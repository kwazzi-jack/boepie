import datetime as dt

from fastmcp import FastMCP
from fastmcp.tools import Tool

from boepie.imaging import RunResult, WSCleanInput, run_wsclean

mcp = FastMCP("Boepie MCP Server")


@mcp.tool
def test_call(name: str) -> str:
    """Greet a user"""
    ts = dt.datetime.strftime(dt.datetime.now(), "%d/%m/%Y, %H:%M:%S")
    return f"Hi, {name}! Here is the timestamp: {ts}"


mcp.add_tool(tool=Tool.from_function(run_wsclean, description="Runs WSClean software"))
