import json
import asyncio
import websockets
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MCPServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.connections = set()

    async def handle_command(self, command):
        """Execute a command and return its output"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "returncode": process.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    async def handle_client(self, websocket, path):
        """Handle individual client connections"""
        try:
            self.connections.add(websocket)
            logger.info(f"New client connected. Total connections: {len(self.connections)}")

            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get("command")
                    
                    if not command:
                        await websocket.send(json.dumps({
                            "error": "No command specified"
                        }))
                        continue

                    logger.info(f"Executing command: {command}")
                    result = await self.handle_command(command)
                    
                    response = {
                        "command": command,
                        "timestamp": datetime.now().isoformat(),
                        "result": result
                    }
                    
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON format"
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "error": str(e)
                    }))

        finally:
            self.connections.remove(websocket)
            logger.info(f"Client disconnected. Remaining connections: {len(self.connections)}")

    async def start(self):
        """Start the MCP server"""
        server = await websockets.serve(self.handle_client, self.host, self.port)
        logger.info(f"MCP Server started on ws://{self.host}:{self.port}")
        await server.wait_closed()

def main():
    server = MCPServer()
    asyncio.get_event_loop().run_until_complete(server.start())
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    main()