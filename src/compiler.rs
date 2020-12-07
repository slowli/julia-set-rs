//! Code shared among backends.

use arithmetic_parser::BinaryOp;

use crate::{function::Evaluated, Function};

const COMPUTE_ARGUMENT: &str = "z";
const VAR_PREFIX: &str = "__var_";
const FN_PREFIX: &str = "complex_";

#[derive(Debug, Clone, Copy)]
pub(crate) struct Compiler {
    complex_ty: &'static str,
    complex_init: &'static str,
}

impl Compiler {
    #[cfg(any(test, feature = "opencl_backend"))]
    pub fn for_ocl() -> Self {
        Self {
            complex_ty: "float2",
            complex_init: "(float2)",
        }
    }

    #[cfg(any(test, feature = "vulkan_backend"))]
    pub fn for_gl() -> Self {
        Self {
            complex_ty: "vec2",
            complex_init: "vec2",
        }
    }

    pub fn compile(self, function: &Function) -> String {
        let mut code = String::new();
        for (var_name, value) in function.assignments() {
            code += &format!("{} {}{} = ", self.complex_ty, VAR_PREFIX, var_name);
            self.compile_expr(&mut code, value);
            code += "; ";
        }

        code += "return ";
        self.compile_expr(&mut code, function.return_value());
        code += ";";
        code
    }

    fn op_function(op: BinaryOp) -> &'static str {
        match op {
            BinaryOp::Mul => "complex_mul",
            BinaryOp::Div => "complex_div",
            BinaryOp::Power => "complex_pow",
            _ => unreachable!(),
        }
    }

    fn compile_expr(self, dest: &mut String, expr: &Evaluated) {
        match expr {
            Evaluated::Variable(name) => {
                if name != COMPUTE_ARGUMENT {
                    dest.push_str(VAR_PREFIX);
                }
                dest.push_str(name);
            }

            Evaluated::Value(val) => {
                dest.push_str(self.complex_init);
                dest.push_str("(");
                dest.push_str(&val.re.to_string());
                dest.push_str(", ");
                dest.push_str(&val.im.to_string());
                dest.push_str(")");
            }

            Evaluated::Negation(inner) => {
                dest.push('-');
                self.compile_expr(dest, inner);
            }

            Evaluated::Binary { op, lhs, rhs } => match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    self.compile_expr(dest, lhs);
                    dest.push(' ');
                    dest.push_str(op.as_str());
                    dest.push(' ');
                    self.compile_expr(dest, rhs);
                }

                _ => {
                    let function_name = Self::op_function(*op);
                    dest.push_str(function_name);
                    dest.push('(');
                    self.compile_expr(dest, lhs);
                    dest.push_str(", ");
                    self.compile_expr(dest, rhs);
                    dest.push(')');
                }
            },

            Evaluated::FunctionCall { name, args } => {
                dest.push_str(FN_PREFIX);
                dest.push_str(name);
                dest.push('(');
                for (i, arg) in args.iter().enumerate() {
                    self.compile_expr(dest, arg);
                    if i + 1 < args.len() {
                        dest.push_str(", ");
                    }
                }
                dest.push(')');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiling_simple_fns() {
        let function = "z*z + 0.2 + 0.5i".parse().unwrap();
        let code = Compiler::for_ocl().compile(&function);
        assert_eq!(code, "return complex_mul(z, z) + (float2)(0.2, 0.5);");
        let code = Compiler::for_gl().compile(&function);
        assert_eq!(code, "return complex_mul(z, z) + vec2(0.2, 0.5);");

        let function = "z^3 * sinh(0.2 + z*z)".parse().unwrap();
        let code = Compiler::for_ocl().compile(&function);
        assert_eq!(
            code,
            "return complex_mul(complex_pow(z, (float2)(3, 0)), \
             complex_sinh(complex_mul(z, z) + (float2)(0.2, 0)));"
        );
        let code = Compiler::for_gl().compile(&function);
        assert_eq!(
            code,
            "return complex_mul(complex_pow(z, vec2(3, 0)), \
             complex_sinh(complex_mul(z, z) + vec2(0.2, 0)));"
        );
    }

    #[test]
    fn complex_function_arg() {
        let function = "sinh(z^2 + 2i * z * -0.5)".parse().unwrap();
        let code = Compiler::for_ocl().compile(&function);
        assert_eq!(
            code,
            "return complex_sinh(complex_pow(z, (float2)(2, 0)) + \
             complex_mul(z, (float2)(0, -1)));"
        );

        let function = "0.7 + cosh(z*z - 0.5i) * z".parse().unwrap();
        let code = Compiler::for_ocl().compile(&function);
        assert_eq!(
            code,
            "return complex_mul(complex_cosh(complex_mul(z, z) + (float2)(0, -0.5)), z) + \
             (float2)(0.7, 0);"
        );
    }

    #[test]
    fn compiling_fn_with_assignment() {
        let function = "c = 0.5 + 0.4i; z*z + c".parse().unwrap();
        let code = Compiler::for_ocl().compile(&function);
        assert_eq!(
            code,
            "float2 __var_c = (float2)(0.5, 0.4); \
             return complex_mul(z, z) + __var_c;"
        );

        let function = "d = sinh(z) * z * 1.1; z*z - 0.5 + d".parse().unwrap();
        let code = Compiler::for_ocl().compile(&function);
        assert_eq!(
            code,
            "float2 __var_d = complex_mul(complex_mul(complex_sinh(z), z), (float2)(1.1, 0)); \
             return complex_mul(z, z) + __var_d + (float2)(-0.5, 0);"
        );
    }
}
