"""RPC client for measured MetaWorld state, actions and synchronized RGB-D."""

from rpent.robots.components.env_client_base import BaseEnvClient


class MetaWorldEnvClient(BaseEnvClient):
    def control(self, operation, *args):
        return self._client.call("env." + operation, args=args, timeout_s=120.0)
