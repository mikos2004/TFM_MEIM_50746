#############################
# Biblioteca
#############################
import asyncio
from fastmcp import Client

async def main():
    # Conecta ao servidor HTTP da calculadora
    client = Client("http://localhost:8003/mcp")
    
    async with client:
        # Verifica se o servidor está à escuta
        await client.ping()
        print(" >>> Servidor conectado com sucesso!")

        # Lista todas as ferramentas disponíveis
        tools = await client.list_tools()
        print(f" >>> Ferramentas disponíveis:")
        for tool in tools:
            print(f" - {tool.name}: {tool.description}")

        # Exemplo: Somar dois números
        result_add = await client.call_tool("add", {"a": 10, "b": 5})
        print(f"\t10 + 5 = {result_add.data}")

        # Exemplo: Multiplicar dois números
        result_multiply = await client.call_tool("multiply", {"a": 4, "b": 3})
        print(f"\t4 * 3 = {result_multiply.data}")

        # Exemplo: Subtrair dois números
        result_subtract = await client.call_tool("subtract", {"a": 20, "b": 7})
        print(f"\t20 - 7 = {result_subtract.data}")

        # Exemplo: Dividir dois números
        try:
            result_divide = await client.call_tool("divide", {"a": 15, "b": 3})
            print(f"\t15 / 3 = {result_divide.data}")
        except Exception as e:
            print(f"\tErro na divisão: {e}")

        # Exemplo: Tentar dividir por zero (para mostrar tratamento de erro)
        try:
            result_divide_zero = await client.call_tool("divide", {"a": 10, "b": 0})
            
            # Verifica se houve erro na resposta
            if result_divide_zero.is_error:
                print(f"\t Erro ao dividir por zero: {result_divide_zero.content[0].text}")
            else:
                print(f"\t 10 / 0 = {result_divide_zero.data}")
                
        except Exception as e:
            print(f"\t Erro ao dividir por zero: {e}")


#############################
# Main
#############################

if __name__ == "__main__":
    asyncio.run(main())