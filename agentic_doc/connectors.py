"""Backward compatibility wrapper for connectors.

This module provides backward compatibility for the connector functionality
that has been moved to the ade-connectors package.
"""

try:
    from ade_connectors import (
        BaseConnector,
        ConnectorConfig,
        LocalConnector,
        LocalConnectorConfig,
        GoogleDriveConnector,
        GoogleDriveConnectorConfig,
        S3Connector,
        S3ConnectorConfig,
        URLConnector,
        URLConnectorConfig,
        create_connector,
    )
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False

    # Define stub classes that raise helpful errors
    class _ConnectorNotAvailable:
        """Stub class for when connectors are not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Connectors are not available. Install them with: "
                "pip install ade-connectors"
            )

    class ConnectorConfig(_ConnectorNotAvailable):
        """Stub for ConnectorConfig."""
        connector_type: str = ""

    class BaseConnector(_ConnectorNotAvailable):
        """Stub for BaseConnector."""
        pass

    class LocalConnectorConfig(ConnectorConfig):
        """Stub for LocalConnectorConfig."""
        pass

    class LocalConnector(BaseConnector):
        """Stub for LocalConnector."""
        pass

    class GoogleDriveConnectorConfig(ConnectorConfig):
        """Stub for GoogleDriveConnectorConfig."""
        pass

    class GoogleDriveConnector(BaseConnector):
        """Stub for GoogleDriveConnector."""
        pass

    class S3ConnectorConfig(ConnectorConfig):
        """Stub for S3ConnectorConfig."""
        pass

    class S3Connector(BaseConnector):
        """Stub for S3Connector."""
        pass

    class URLConnectorConfig(ConnectorConfig):
        """Stub for URLConnectorConfig."""
        pass

    class URLConnector(BaseConnector):
        """Stub for URLConnector."""
        pass

    def create_connector(*args, **kwargs):
        """Stub for create_connector."""
        raise ImportError(
            "Connectors are not available. Install them with: "
            "pip install ade-connectors"
        )

# Export for backward compatibility
__all__ = [
    "BaseConnector",
    "ConnectorConfig",
    "LocalConnector",
    "LocalConnectorConfig",
    "GoogleDriveConnector",
    "GoogleDriveConnectorConfig",
    "S3Connector",
    "S3ConnectorConfig",
    "URLConnector",
    "URLConnectorConfig",
    "create_connector",
    "CONNECTORS_AVAILABLE",
]