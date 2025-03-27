from typing import Callable, Optional

import toolz
from mcp.server.fastmcp import FastMCP

from xorq.flight import FlightServer, FlightUrl


class FlightMCPServer:
    def __init__(
        self,
        name: str,
        flight_port: int = 8818,
    ):
        self.name = name
        self.port = flight_port
        self.mcp = FastMCP(name)

        self.flight_server = None
        self.client = None

        self.udxfs = {}
        self.schemas = {}
        self.exchange_functions = {}

    def start_flight_server(self) -> FlightServer:
        if self.flight_server:
            return self.flight_server

        try:
            self.flight_server = FlightServer(
                FlightUrl(port=self.port), exchangers=list(self.udxfs.values())
            )

            self.flight_server.serve()

            self.client = self.flight_server.client

            for udxf_name, udxf in self.udxfs.items():
                self.exchange_functions[udxf_name] = toolz.curry(
                    self.client.do_exchange, udxf.command
                )

            return self.flight_server
        except Exception:
            raise

    def create_mcp_tool(
        self,
        udxf,
        input_mapper: Callable,
        tool_name: Optional[str] = None,
        description: Optional[str] = None,
        output_mapper: Optional[Callable] = None,
    ):
        udxf_command = udxf.command
        tool_name = tool_name or udxf_command

        self.udxfs[udxf_command] = udxf

        if not self.flight_server:
            self.start_flight_server()

        do_exchange = self.exchange_functions.get(udxf_command)

        if output_mapper is None:

            def default_output_mapper(result_df):
                if len(result_df) > 0:
                    return result_df.to_string()
                return "No results"

            actual_output_mapper = default_output_mapper
        else:
            actual_output_mapper = output_mapper

        @self.mcp.tool(name=tool_name, description=description)
        async def wrapper(**kwargs):
            try:
                input_data = input_mapper(**kwargs)
                _, result = do_exchange(input_data.to_pyarrow_batches())
                result_df = result.read_pandas()
                output = actual_output_mapper(result_df)
                return output
            except Exception as e:
                return f"Error executing tool: {str(e)}"

        return wrapper

    def run(self, transport: str = "stdio"):
        if not self.flight_server:
            self.start_flight_server()
        try:
            self.mcp.run(transport=transport)
        except Exception:
            raise

    def stop(self):
        # TODO
        pass
