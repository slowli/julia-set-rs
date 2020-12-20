use arithmetic_parser::{
    grammars::{Features, NumGrammar, Parse, Untyped},
    BinaryOp, Block, Expr, Lvalue, Spanned, SpannedExpr, Statement, UnaryOp,
};
use num_complex::Complex32;
use thiserror::Error;

use std::{collections::HashSet, error::Error, fmt, iter, mem, ops, str::FromStr};

/// Error associated with creating a [`Function`].
#[derive(Debug)]
#[cfg_attr(
    docsrs,
    doc(cfg(any(
        feature = "dyn_cpu_backend",
        feature = "opencl_backend",
        feature = "vulkan_backend"
    )))
)]
pub struct FnError {
    fragment: String,
    line: u32,
    column: usize,
    source: ErrorSource,
}

#[derive(Debug)]
enum ErrorSource {
    Parse(String),
    Eval(EvalError),
}

impl fmt::Display for ErrorSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(err) => write!(formatter, "[PARSE] {}", err),
            Self::Eval(err) => write!(formatter, "[EVAL] {}", err),
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum EvalError {
    #[error("Last statement in function body is not an expression")]
    NoReturn,
    #[error("Useless expression")]
    UselessExpr,
    #[error("Cannot redefine variable")]
    RedefinedVar,
    #[error("Undefined variable")]
    UndefinedVar,
    #[error("Undefined function")]
    UndefinedFn,
    #[error("Function call has bogus arity")]
    FnArity,
    #[error("Unsupported language construct")]
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum UnaryFunction {
    Arg,
    Sqrt,
    Exp,
    Log,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
}

impl UnaryFunction {
    #[cfg(any(feature = "opencl_backend", feature = "vulkan_backend"))]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Arg => "arg",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
        }
    }

    #[cfg(feature = "dyn_cpu_backend")]
    pub fn eval(self, arg: Complex32) -> Complex32 {
        match self {
            Self::Arg => Complex32::new(arg.arg(), 0.0),
            Self::Sqrt => arg.sqrt(),
            Self::Exp => arg.exp(),
            Self::Log => arg.ln(),
            Self::Sinh => arg.sinh(),
            Self::Cosh => arg.cosh(),
            Self::Tanh => arg.tanh(),
            Self::Asinh => arg.asinh(),
            Self::Acosh => arg.acosh(),
            Self::Atanh => arg.atanh(),
        }
    }
}

impl FromStr for UnaryFunction {
    type Err = EvalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "arg" => Ok(Self::Arg),
            "sqrt" => Ok(Self::Sqrt),
            "exp" => Ok(Self::Exp),
            "log" => Ok(Self::Log),
            "sinh" => Ok(Self::Sinh),
            "cosh" => Ok(Self::Cosh),
            "tanh" => Ok(Self::Tanh),
            "asinh" => Ok(Self::Asinh),
            "acosh" => Ok(Self::Acosh),
            "atanh" => Ok(Self::Atanh),
            _ => Err(EvalError::UndefinedFn),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Evaluated {
    Value(Complex32),
    Variable(String),
    Negation(Box<Evaluated>),
    Binary {
        op: BinaryOp,
        lhs: Box<Evaluated>,
        rhs: Box<Evaluated>,
    },
    FunctionCall {
        function: UnaryFunction,
        arg: Box<Evaluated>,
    },
}

impl Evaluated {
    fn is_commutative(op: BinaryOp) -> bool {
        matches!(op, BinaryOp::Add | BinaryOp::Mul)
    }

    fn is_commutative_pair(op: BinaryOp, other_op: BinaryOp) -> bool {
        op.priority() == other_op.priority() && op != BinaryOp::Power
    }

    fn fold(mut op: BinaryOp, mut lhs: Self, mut rhs: Self) -> Self {
        // First, check if the both operands are values. In this case, we can eagerly compute
        // the resulting value.
        if let (Self::Value(lhs_val), Self::Value(rhs_val)) = (&lhs, &rhs) {
            return Self::Value(match op {
                BinaryOp::Add => *lhs_val + *rhs_val,
                BinaryOp::Sub => *lhs_val - *rhs_val,
                BinaryOp::Mul => *lhs_val * *rhs_val,
                BinaryOp::Div => *lhs_val / *rhs_val,
                BinaryOp::Power => lhs_val.powc(*rhs_val),
                _ => unreachable!(),
            });
        }

        if let Self::Value(val) = rhs {
            // Convert an RHS value to use a commutative op (e.g., `+` instead of `-`).
            // This will come in handy during later transforms.
            //
            // For example, this will transform `z - 1` into `z + -1`.
            match op {
                BinaryOp::Sub => {
                    op = BinaryOp::Add;
                    rhs = Self::Value(-val);
                }
                BinaryOp::Div => {
                    op = BinaryOp::Mul;
                    rhs = Self::Value(1.0 / val);
                }
                _ => { /* do nothing */ }
            }
        } else if let Self::Value(_) = lhs {
            // Swap LHS and RHS to move the value to the right.
            //
            // For example, this will transform `1 + z` into `z + 1`.
            if Self::is_commutative(op) {
                mem::swap(&mut lhs, &mut rhs);
            }
        }

        if let Self::Binary {
            op: inner_op,
            rhs: inner_rhs,
            ..
        } = &mut lhs
        {
            if Self::is_commutative_pair(*inner_op, op) {
                if let Self::Value(inner_val) = **inner_rhs {
                    if let Self::Value(val) = rhs {
                        // Make the following replacement:
                        //
                        //    op             op
                        //   /  \           /  \
                        //  op  c   ---->  a  b op c
                        // /  \
                        // a  b
                        let new_rhs = match op {
                            BinaryOp::Add => inner_val + val,
                            BinaryOp::Mul => inner_val * val,
                            _ => unreachable!(),
                            // ^-- We've replaced '-' and '/' `op`s previously.
                        };

                        *inner_rhs = Box::new(Self::Value(new_rhs));
                        return lhs;
                    } else {
                        // Switch `inner_rhs` and `rhs`, moving a `Value` to the right.
                        // For example, this will replace `z + 1 - z^2` to `z - z^2 + 1`.
                        mem::swap(&mut rhs, inner_rhs);
                        mem::swap(&mut op, inner_op);
                    }
                }
            }
        }

        Self::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

impl ops::Neg for Evaluated {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Value(val) => Self::Value(-val),
            Self::Negation(inner) => *inner,
            other => Self::Negation(Box::new(other)),
        }
    }
}

impl FnError {
    fn parse(source: &arithmetic_parser::Error<'_>) -> Self {
        let column = source.span().get_column();
        Self {
            fragment: (*source.span().fragment()).to_owned(),
            line: source.span().location_line(),
            column,
            source: ErrorSource::Parse(source.kind().to_string()),
        }
    }

    fn eval<T>(span: &Spanned<'_, T>, source: EvalError) -> Self {
        let column = span.get_column();
        Self {
            fragment: (*span.fragment()).to_owned(),
            line: span.location_line(),
            column,
            source: ErrorSource::Eval(source),
        }
    }
}

impl fmt::Display for FnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}: {}", self.line, self.column, self.source)?;
        if formatter.alternate() {
            formatter.write_str("\n")?;
            formatter.pad(&self.fragment)?;
        }
        Ok(())
    }
}

impl Error for FnError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.source {
            ErrorSource::Eval(e) => Some(e),
            _ => None,
        }
    }
}

type FnGrammarBase = Untyped<NumGrammar<Complex32>>;

#[derive(Debug, Clone, Copy)]
struct FnGrammar;

impl Parse for FnGrammar {
    type Base = FnGrammarBase;
    const FEATURES: Features = Features::empty();
}

#[derive(Debug)]
pub(crate) struct Context {
    variables: HashSet<String>,
}

impl Context {
    pub(crate) fn new(arg_name: &str) -> Self {
        Self {
            variables: iter::once(arg_name.to_owned()).collect(),
        }
    }

    fn process(
        &mut self,
        block: &Block<'_, FnGrammarBase>,
        total_span: Spanned<'_>,
    ) -> Result<Function, FnError> {
        let mut assignments = vec![];
        for statement in &block.statements {
            match &statement.extra {
                Statement::Assignment { lhs, rhs } => {
                    let variable_name = match lhs.extra {
                        Lvalue::Variable { .. } => *lhs.fragment(),
                        _ => unreachable!("Tuples are disabled in parser"),
                    };

                    if self.variables.contains(variable_name) {
                        let err = FnError::eval(lhs, EvalError::RedefinedVar);
                        return Err(err);
                    }

                    // Evaluate the RHS.
                    let value = self.eval_expr(rhs)?;
                    self.variables.insert(variable_name.to_owned());
                    assignments.push((variable_name.to_owned(), value));
                }

                Statement::Expr(_) => {
                    return Err(FnError::eval(&statement, EvalError::UselessExpr));
                }

                _ => return Err(FnError::eval(&statement, EvalError::Unsupported)),
            }
        }

        let return_value = block
            .return_value
            .as_ref()
            .ok_or_else(|| FnError::eval(&total_span, EvalError::NoReturn))?;
        let value = self.eval_expr(return_value)?;
        assignments.push((String::new(), value));

        Ok(Function { assignments })
    }

    fn eval_expr(&self, expr: &SpannedExpr<'_, FnGrammarBase>) -> Result<Evaluated, FnError> {
        match &expr.extra {
            Expr::Variable => {
                let var_name = *expr.fragment();
                self.variables
                    .get(var_name)
                    .ok_or_else(|| FnError::eval(expr, EvalError::UndefinedVar))?;

                Ok(Evaluated::Variable(var_name.to_owned()))
            }
            Expr::Literal(lit) => Ok(Evaluated::Value(*lit)),

            Expr::Unary { op, inner } => match op.extra {
                UnaryOp::Neg => Ok(-self.eval_expr(inner)?),
                _ => Err(FnError::eval(op, EvalError::Unsupported)),
            },

            Expr::Binary { lhs, op, rhs } => {
                let lhs_value = self.eval_expr(lhs)?;
                let rhs_value = self.eval_expr(rhs)?;

                Ok(match op.extra {
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Power => Evaluated::fold(op.extra, lhs_value, rhs_value),
                    _ => {
                        return Err(FnError::eval(op, EvalError::Unsupported));
                    }
                })
            }

            Expr::Function { name, args } => {
                let fn_name = *name.fragment();
                let function: UnaryFunction =
                    fn_name.parse().map_err(|e| FnError::eval(name, e))?;

                if args.len() != 1 {
                    return Err(FnError::eval(expr, EvalError::FnArity));
                }

                Ok(Evaluated::FunctionCall {
                    function,
                    arg: Box::new(self.eval_expr(&args[0])?),
                })
            }

            Expr::FnDefinition(_) | Expr::Block(_) | Expr::Tuple(_) | Expr::Method { .. } => {
                unreachable!("Disabled in parser")
            }

            _ => Err(FnError::eval(expr, EvalError::Unsupported)),
        }
    }
}

/// Parsed complex-valued function of a single variable.
///
/// A `Function` instance can be created using [`FromStr`] trait. A function must use `z`
/// as the (only) argument. A function may use arithmetic operations (`+`, `-`, `*`, `/`, `^`)
/// and/or predefined unary functions:
///
/// - General functions: `arg`, `sqrt`, `exp`, `log`
/// - Hyperbolic trigonometry: `sinh`, `cosh`, `tanh`
/// - Inverse hyperbolic trigonometry: `asinh`, `acosh`, `atanh`
///
/// A function may define local variable assignment(s). The assignment syntax is similar to Python
/// (or Rust, just without the `let` keyword): variable name followed by `=` and then by
/// the arithmetic expression. Assignments must be separated by semicolons `;`. As in Rust,
/// the last expression in function body is its return value.
///
/// # Examples
///
/// ```
/// # use julia_set::Function;
/// # fn main() -> anyhow::Result<()> {
/// let function: Function = "z * z - 0.5".parse()?;
/// let fn_with_calls: Function = "0.8 * z + z / atanh(z ^ -4)".parse()?;
/// let fn_with_vars: Function = "c = -0.5 + 0.4i; z * z + c".parse()?;
/// # Ok(())
/// # }
/// ```
#[cfg_attr(
    docsrs,
    doc(cfg(any(
        feature = "dyn_cpu_backend",
        feature = "opencl_backend",
        feature = "vulkan_backend"
    )))
)]
#[derive(Debug, Clone)]
pub struct Function {
    assignments: Vec<(String, Evaluated)>,
}

impl Function {
    pub(crate) fn assignments(&self) -> impl Iterator<Item = (&str, &Evaluated)> + '_ {
        self.assignments.iter().filter_map(|(name, value)| {
            if name.is_empty() {
                None
            } else {
                Some((name.as_str(), value))
            }
        })
    }

    pub(crate) fn return_value(&self) -> &Evaluated {
        &self.assignments.last().unwrap().1
    }
}

impl FromStr for Function {
    type Err = FnError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let statements = FnGrammar::parse_statements(s).map_err(|e| FnError::parse(&e))?;
        let body_span = Spanned::from_str(s, ..);
        Context::new("z").process(&statements, body_span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z_square() -> Box<Evaluated> {
        Box::new(Evaluated::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(Evaluated::Variable("z".to_owned())),
            rhs: Box::new(Evaluated::Variable("z".to_owned())),
        })
    }

    #[test]
    fn simple_function() {
        let function: Function = "z*z + (0.77 - 0.2i)".parse().unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Add,
            lhs: z_square(),
            rhs: Box::new(Evaluated::Value(Complex32::new(0.77, -0.2))),
        };
        assert_eq!(function.assignments, vec![(String::new(), expected_expr)]);
    }

    #[test]
    fn simple_function_with_rewrite_rules() {
        let function: Function = "z / 0.25 - 0.1i + (0.77 - 0.1i)".parse().unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(Evaluated::Binary {
                op: BinaryOp::Mul,
                lhs: Box::new(Evaluated::Variable("z".to_owned())),
                rhs: Box::new(Evaluated::Value(Complex32::new(4.0, 0.0))),
            }),
            rhs: Box::new(Evaluated::Value(Complex32::new(0.77, -0.2))),
        };
        assert_eq!(function.assignments, vec![(String::new(), expected_expr)]);
    }

    #[test]
    fn function_with_several_rewrite_rules() {
        let function: Function = "z + 0.1 - z*z + 0.3i".parse().unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(Evaluated::Binary {
                op: BinaryOp::Sub,
                lhs: Box::new(Evaluated::Variable("z".to_owned())),
                rhs: z_square(),
            }),
            rhs: Box::new(Evaluated::Value(Complex32::new(0.1, 0.3))),
        };
        assert_eq!(function.assignments, vec![(String::new(), expected_expr)]);
    }

    #[test]
    fn simple_function_with_mul_rewrite_rules() {
        let function: Function = "sinh(z - 5) / 4. * 6i".parse().unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(Evaluated::FunctionCall {
                function: UnaryFunction::Sinh,
                arg: Box::new(Evaluated::Binary {
                    op: BinaryOp::Add,
                    lhs: Box::new(Evaluated::Variable("z".to_owned())),
                    rhs: Box::new(Evaluated::Value(Complex32::new(-5.0, 0.0))),
                }),
            }),
            rhs: Box::new(Evaluated::Value(Complex32::new(0.0, 1.5))),
        };
        assert_eq!(function.assignments, vec![(String::new(), expected_expr)]);
    }

    #[test]
    fn simple_function_with_redistribution() {
        let function: Function = "0.5 + sinh(z) - 0.2i".parse().unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(Evaluated::FunctionCall {
                function: UnaryFunction::Sinh,
                arg: Box::new(Evaluated::Variable("z".to_owned())),
            }),
            rhs: Box::new(Evaluated::Value(Complex32::new(0.5, -0.2))),
        };
        assert_eq!(function.assignments, vec![(String::new(), expected_expr)]);
    }

    #[test]
    fn function_with_assignments() {
        let function: Function = "c = 0.5 - 0.2i; z*z + c".parse().unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Add,
            lhs: z_square(),
            rhs: Box::new(Evaluated::Variable("c".to_owned())),
        };

        assert_eq!(
            function.assignments,
            vec![
                ("c".to_owned(), Evaluated::Value(Complex32::new(0.5, -0.2))),
                (String::new(), expected_expr),
            ]
        );
    }
}
