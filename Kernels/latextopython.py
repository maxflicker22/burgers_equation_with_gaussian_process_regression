from latex2sympy2 import latex2sympy
import sympy as sp

# Diese zwei variablen muss man ändern! und die replace_variables function!!!
filename = "generated_function_dk4dx2dy2dlogsigma"
latex_expression = r"\frac{27 \left(2\mathrm{e}^{2{\sigma}} + \left(2x^{2} \mathrm{e}^{\eta} + 3\right) \mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \mathrm{e}^{{\sigma} + 2{\eta}} \left(y \left(-8x\mathrm{e}^{3{\sigma} + {\eta}} - 18x\mathrm{e}^{2{\sigma} + {\eta}} - 15x\mathrm{e}^{{\sigma} + {\eta}} - 5x\mathrm{e}^{\eta}\right) + y^{2} \left(4\mathrm{e}^{3{\sigma} + {\eta}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 6\mathrm{e}^{\eta}\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 3\mathrm{e}^{\eta}\right) \mathrm{e}^{\sigma}\right) + y^{3} \left(-8x\mathrm{e}^{2{\sigma} + 2{\eta}} - 10x\mathrm{e}^{{\sigma} + 2{\eta}} - 5x\mathrm{e}^{2{\eta}}\right) + y^{4} \left(4\mathrm{e}^{2{\sigma} + 2{\eta}} + 4\mathrm{e}^{{\sigma} + 2{\eta}}\right) + \left(4x^{2} \mathrm{e}^{\eta} - 2\right) \mathrm{e}^{3{\sigma}} + \left(8x^{2} \mathrm{e}^{\eta} - 3\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{\eta} - 1\right) \mathrm{e}^{\sigma}\right)}{\pi \left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{11}{2}} \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{9}{2}} \left(1 - \frac{\left(\mathrm{e}^{\sigma} + xy\mathrm{e}^{\eta}\right)^{2}}{\left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)}\right)^{\frac{7}{2}}} + \frac{27 \left(2\mathrm{e}^{2{\sigma}} + \left(2x^{2} \mathrm{e}^{\eta} + 3\right) \mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \mathrm{e}^{{\sigma} + 2{\eta}} \left(y \left(-8x\mathrm{e}^{3{\sigma} + {\eta}} - 18x\mathrm{e}^{2{\sigma} + {\eta}} - 15x\mathrm{e}^{{\sigma} + {\eta}} - 5x\mathrm{e}^{\eta}\right) + y^{2} \left(4\mathrm{e}^{3{\sigma} + {\eta}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 6\mathrm{e}^{\eta}\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 3\mathrm{e}^{\eta}\right) \mathrm{e}^{\sigma}\right) + y^{3} \left(-8x\mathrm{e}^{2{\sigma} + 2{\eta}} - 10x\mathrm{e}^{{\sigma} + 2{\eta}} - 5x\mathrm{e}^{2{\eta}}\right) + y^{4} \left(4\mathrm{e}^{2{\sigma} + 2{\eta}} + 4\mathrm{e}^{{\sigma} + 2{\eta}}\right) + \left(4x^{2} \mathrm{e}^{\eta} - 2\right) \mathrm{e}^{3{\sigma}} + \left(8x^{2} \mathrm{e}^{\eta} - 3\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{\eta} - 1\right) \mathrm{e}^{\sigma}\right)}{\pi \left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{9}{2}} \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{11}{2}} \left(1 - \frac{\left(\mathrm{e}^{\sigma} + xy\mathrm{e}^{\eta}\right)^{2}}{\left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)}\right)^{\frac{7}{2}}} - \frac{6\mathrm{e}^{2{\eta}} \left(4\mathrm{e}^{2{\sigma}} + \left(2x^{2} \mathrm{e}^{\eta} + 3\right) \mathrm{e}^{\sigma}\right) \left(y \left(-8x\mathrm{e}^{3{\sigma} + {\eta}} - 18x\mathrm{e}^{2{\sigma} + {\eta}} - 15x\mathrm{e}^{{\sigma} + {\eta}} - 5x\mathrm{e}^{\eta}\right) + y^{2} \left(4\mathrm{e}^{3{\sigma} + {\eta}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 6\mathrm{e}^{\eta}\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 3\mathrm{e}^{\eta}\right) \mathrm{e}^{\sigma}\right) + y^{3} \left(-8x\mathrm{e}^{2{\sigma} + 2{\eta}} - 10x\mathrm{e}^{{\sigma} + 2{\eta}} - 5x\mathrm{e}^{2{\eta}}\right) + y^{4} \left(4\mathrm{e}^{2{\sigma} + 2{\eta}} + 4\mathrm{e}^{{\sigma} + 2{\eta}}\right) + \left(4x^{2} \mathrm{e}^{\eta} - 2\right) \mathrm{e}^{3{\sigma}} + \left(8x^{2} \mathrm{e}^{\eta} - 3\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{\eta} - 1\right) \mathrm{e}^{\sigma}\right)}{\pi \left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{9}{2}} \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{9}{2}} \left(1 - \frac{\left(\mathrm{e}^{\sigma} + xy\mathrm{e}^{\eta}\right)^{2}}{\left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)}\right)^{\frac{7}{2}}} + \frac{21\mathrm{e}^{2{\eta}} \left(\frac{\mathrm{e}^{\sigma} \left(\mathrm{e}^{\sigma} + xy\mathrm{e}^{\eta}\right)^{2}}{\left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right)^{2} \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)} - \frac{2\mathrm{e}^{\sigma} \left(\mathrm{e}^{\sigma} + xy\mathrm{e}^{\eta}\right)}{\left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)} + \frac{\mathrm{e}^{\sigma} \left(\mathrm{e}^{\sigma} + xy\mathrm{e}^{\eta}\right)^{2}}{\left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)^{2}}\right) \left(2\mathrm{e}^{2{\sigma}} + \left(2x^{2} \mathrm{e}^{\eta} + 3\right) \mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(y \left(-8x\mathrm{e}^{3{\sigma} + {\eta}} - 18x\mathrm{e}^{2{\sigma} + {\eta}} - 15x\mathrm{e}^{{\sigma} + {\eta}} - 5x\mathrm{e}^{\eta}\right) + y^{2} \left(4\mathrm{e}^{3{\sigma} + {\eta}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 6\mathrm{e}^{\eta}\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 3\mathrm{e}^{\eta}\right) \mathrm{e}^{\sigma}\right) + y^{3} \left(-8x\mathrm{e}^{2{\sigma} + 2{\eta}} - 10x\mathrm{e}^{{\sigma} + 2{\eta}} - 5x\mathrm{e}^{2{\eta}}\right) + y^{4} \left(4\mathrm{e}^{2{\sigma} + 2{\eta}} + 4\mathrm{e}^{{\sigma} + 2{\eta}}\right) + \left(4x^{2} \mathrm{e}^{\eta} - 2\right) \mathrm{e}^{3{\sigma}} + \left(8x^{2} \mathrm{e}^{\eta} - 3\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{\eta} - 1\right) \mathrm{e}^{\sigma}\right)}{\pi \left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{9}{2}} \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{9}{2}} \left(1 - \frac{\left(\mathrm{e}^{\sigma} + xy\mathrm{e}^{\eta}\right)^{2}}{\left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)}\right)^{\frac{9}{2}}} - \frac{6\mathrm{e}^{2{\eta}} \left(2\mathrm{e}^{2{\sigma}} + \left(2x^{2} \mathrm{e}^{\eta} + 3\right) \mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(y \left(-24x\mathrm{e}^{3{\sigma} + {\eta}} - 36x\mathrm{e}^{2{\sigma} + {\eta}} - 15x\mathrm{e}^{{\sigma} + {\eta}}\right) + y^{2} \left(12\mathrm{e}^{3{\sigma} + {\eta}} + 2 \left(4x^{2} \mathrm{e}^{2{\eta}} + 6\mathrm{e}^{\eta}\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{2{\eta}} + 3\mathrm{e}^{\eta}\right) \mathrm{e}^{\sigma}\right) + y^{3} \left(-16x\mathrm{e}^{2{\sigma} + 2{\eta}} - 10x\mathrm{e}^{{\sigma} + 2{\eta}}\right) + y^{4} \left(8\mathrm{e}^{2{\sigma} + 2{\eta}} + 4\mathrm{e}^{{\sigma} + 2{\eta}}\right) + 3 \left(4x^{2} \mathrm{e}^{\eta} - 2\right) \mathrm{e}^{3{\sigma}} + 2 \left(8x^{2} \mathrm{e}^{\eta} - 3\right) \mathrm{e}^{2{\sigma}} + \left(4x^{2} \mathrm{e}^{\eta} - 1\right) \mathrm{e}^{\sigma}\right)}{\pi \left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{9}{2}} \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)^{\frac{9}{2}} \left(1 - \frac{\left(\mathrm{e}^{\sigma} + xy\mathrm{e}^{\eta}\right)^{2}}{\left(\mathrm{e}^{\sigma} + x^{2} \mathrm{e}^{\eta} + 1\right) \left(\mathrm{e}^{\sigma} + y^{2} \mathrm{e}^{\eta} + 1\right)}\right)^{\frac{7}{2}}}"

def convert_latex_to_sympy(latex_expression):
    try:
        # Convert the LaTeX expression to a SymPy expression
        sympy_expr = latex2sympy(latex_expression)
        return sympy_expr
    except Exception as e:
        return str(e)


def replace_variables(sympy_expr):
    # Replace variables in the SymPy expression
    replacements = {
        sp.Symbol('eta'): sp.Symbol('logtheta'),
        sp.Symbol('sigma'): sp.Symbol('logsigma')
    }
    return sympy_expr.subs(replacements)


def generate_python_function(sympy_expr, func_name=filename):
    # Get the variables used in the SymPy expression
    variables = sympy_expr.free_symbols
    variables = sorted(variables, key=lambda x: x.name)  # Sort for consistency

    # Generate function signature
    func_signature = f"def {func_name}({', '.join(map(str, variables))}):\n"

    # Generate the return statement
    return_statement = f"    return {sympy_expr}\n"

    # Combine them into the function code
    func_code = func_signature + return_statement
    return func_code


def save_to_file(code, filename=f"{filename}.py"):
    with open(filename, "w") as f:
        f.write(code)


if __name__ == "__main__":
    # Placeholder LaTeX expression
    latex_expression = latex_expression

    # Convert LaTeX to SymPy
    sympy_expr = convert_latex_to_sympy(latex_expression)

    # Replace variables in the SymPy expression
    sympy_expr_replaced = replace_variables(sympy_expr)

    # Generate Python function code
    python_function_code = generate_python_function(sympy_expr_replaced)

    # Save to a Python script
    save_to_file(python_function_code)

    print(f"Generated function saved to '{filename}.py'")
    print(f"Function code:\n{python_function_code}")
