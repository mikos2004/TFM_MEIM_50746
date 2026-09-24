#############################
# Bibliotecas
#############################
from fastmcp import FastMCP

mcp = FastMCP(name = "Calculator")

# decorator para declarar funções
@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    args: a (float): The first number.
          b (float): The second number.

    returns: float: The product of the two numbers.
    """
    return a * b

@mcp.tool(
    name = "add",
    description = "Add two numbers", # o agente lê o topo da docstring, não este campo
    tags = {"math", "arithmetic"}
)
def add(a: float, b: float) -> float:
    """Add two numbers.

    args: a (float): The first number.
          b (float): The second number.

    returns: float: The sum of the two numbers.
    """
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract two numbers.

    args: a (float): The first number.
          b (float): The second number.

    returns: float: The difference of the two numbers.
    """
    return a - b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers.

    args: a (float): The first number.
          b (float): The second number.

    returns: float: The quotient of the two numbers.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

#############################
# Main
#############################

if __name__ == "__main__":
    transport = "http"  # "stdio"

    if transport == "stdio":
        mcp.run(transport="stdio")  # STDIO
    elif transport == "http":
        mcp.run(transport="http", host="localhost", port=8003)
    else:
        raise ValueError(f"Unknow transport: {transport}")
    