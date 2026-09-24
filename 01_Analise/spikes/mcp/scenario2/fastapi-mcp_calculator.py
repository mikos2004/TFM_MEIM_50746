#############################
# Bibliotecas
#############################
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

# 1- CRIAR A API
app = FastAPI(title="Calculator API")

@app.post("/multiply")
def multiply_number(a: float, b: float):
    """
    Multiplies two number and returns the result.
    """
    result =  a * b
    return {"result": result}

@app.post("/sum")
def add_numbers(a: float, b: float):
    """
    Sums two numbers and returns the result.
    """
    result = a + b
    return {"result": result}

@app.post("/subtract")
def subtract_numbers(a: float, b: float):
    """
    Subtracts two numbers and returns the result.
    """
    result = a - b
    return {"result": result}

@app.post("/divide")
def divide_numbers(a: float, b: float):
    """
    Divides two numbers and returns the result.
    """
    if b == 0:
        return {"error": "Division by zero is not allowed"}
    result = a / b
    return {"result": result}

# 2 - CONVERTER PARA MCP
mcp = FastApiMCP(app, name="Calculator MCP")
mcp.mount_http()



#############################
# Main
#############################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002) # default é 8080

