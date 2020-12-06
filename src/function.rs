use arithmetic_parser::{
    grammars::{Features, NumGrammar, Parse, Untyped},
    BinaryOp, Block, Expr, Lvalue, Spanned, SpannedExpr, Statement, UnaryOp,
};
use num_complex::Complex32;
use thiserror::Error;

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt, iter, mem, ops,
};

#[derive(Debug)]
pub struct FnError {
    fragment: String,
    line: u32,
    column: usize,
    source: ErrorSource,
}

#[derive(Debug)]
pub enum ErrorSource {
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
pub enum EvalError {
    #[error("Function body cannot be empty")]
    Empty,
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

#[derive(Debug, Clone, PartialEq)]
pub enum Evaluated {
    Value(Complex32),
    Variable(String),
    Negation(Box<Evaluated>),
    Binary {
        op: BinaryOp,
        lhs: Box<Evaluated>,
        rhs: Box<Evaluated>,
    },
    FunctionCall {
        name: String,
        args: Vec<Evaluated>,
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
    fn parse(source: arithmetic_parser::Error<'_>) -> Self {
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
pub struct Context {
    variables: HashSet<String>,
    functions: HashMap<String, usize>,
}

impl Context {
    pub(crate) fn new<'a>(
        builtin_functions: impl Iterator<Item = (&'a str, usize)>,
        arg_name: &str,
    ) -> Self {
        Self {
            variables: iter::once(arg_name.to_owned()).collect(),
            functions: builtin_functions
                .map(|(name, arity)| (name.to_owned(), arity))
                .collect(),
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
                let arity = if let Some(arity) = self.functions.get(fn_name) {
                    *arity
                } else {
                    return Err(FnError::eval(name, EvalError::UndefinedFn));
                };

                if args.len() != arity {
                    return Err(FnError::eval(expr, EvalError::FnArity));
                }

                let arg_values = args.iter().map(|arg| self.eval_expr(arg));
                Ok(Evaluated::FunctionCall {
                    name: fn_name.to_owned(),
                    args: arg_values.collect::<Result<_, FnError>>()?,
                })
            }

            Expr::FnDefinition(_) | Expr::Block(_) | Expr::Tuple(_) | Expr::Method { .. } => {
                unreachable!("Disabled in parser")
            }

            _ => Err(FnError::eval(expr, EvalError::Unsupported)),
        }
    }
}

const FUNCTIONS: &[&str] = &[
    "arg", "sqrt", "exp", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
];

#[derive(Debug, Clone)]
pub struct Function {
    assignments: Vec<(String, Evaluated)>,
}

impl Function {
    pub fn new(body: &str) -> Result<Self, FnError> {
        let statements = FnGrammar::parse_statements(body).map_err(FnError::parse)?;
        let body_span = Spanned::from_str(body, ..);
        let builtin_functions = FUNCTIONS.iter().copied().map(|name| (name, 1));
        Context::new(builtin_functions, "z").process(&statements, body_span)
    }

    pub fn assignments(&self) -> impl Iterator<Item = (&str, &Evaluated)> + '_ {
        self.assignments.iter().filter_map(|(name, value)| {
            if name.is_empty() {
                None
            } else {
                Some((name.as_str(), value))
            }
        })
    }

    pub fn return_value(&self) -> &Evaluated {
        &self.assignments.last().unwrap().1
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
        let function = Function::new("z*z + (0.77 - 0.2i)").unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Add,
            lhs: z_square(),
            rhs: Box::new(Evaluated::Value(Complex32::new(0.77, -0.2))),
        };
        assert_eq!(function.assignments, vec![(String::new(), expected_expr)]);
    }

    #[test]
    fn simple_function_with_rewrite_rules() {
        let function = Function::new("z / 0.25 - 0.1i + (0.77 - 0.1i)").unwrap();
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
        let function = Function::new("z + 0.1 - z*z + 0.3i").unwrap();
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
        let function = Function::new("sinh(z - 5) / 4. * 6i").unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(Evaluated::FunctionCall {
                name: "sinh".to_owned(),
                args: vec![Evaluated::Binary {
                    op: BinaryOp::Add,
                    lhs: Box::new(Evaluated::Variable("z".to_owned())),
                    rhs: Box::new(Evaluated::Value(Complex32::new(-5.0, 0.0))),
                }],
            }),
            rhs: Box::new(Evaluated::Value(Complex32::new(0.0, 1.5))),
        };
        assert_eq!(function.assignments, vec![(String::new(), expected_expr)]);
    }

    #[test]
    fn simple_function_with_redistribution() {
        let function = Function::new("0.5 + sinh(z) - 0.2i").unwrap();
        let expected_expr = Evaluated::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(Evaluated::FunctionCall {
                name: "sinh".to_owned(),
                args: vec![Evaluated::Variable("z".to_owned())],
            }),
            rhs: Box::new(Evaluated::Value(Complex32::new(0.5, -0.2))),
        };
        assert_eq!(function.assignments, vec![(String::new(), expected_expr)]);
    }

    #[test]
    fn function_with_assignments() {
        const BODY: &str = "c = 0.5 - 0.2i; z*z + c";
        let function = Function::new(BODY).unwrap();
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
