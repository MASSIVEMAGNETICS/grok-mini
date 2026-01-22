"""
self_agents.py

Agent introspection and audit scaffolding for autonomous behaviors.

Provides:
- AgentAction: dataclass for recording agent actions with metadata
- AgentAuditor: logging and approval system for autonomous actions
- safe_execute: wrapper for executing agent actions with approval gates

Security model:
- All network and system access requires opt-in via environment flags
- Operator approval required for sensitive operations
- Full audit trail of all autonomous actions
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentAction:
    """Record of an agent action with metadata for auditing."""
    action_type: str  # e.g., "network_request", "file_write", "code_execution"
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    approved: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

class AgentAuditor:
    """
    Auditing and approval system for agent actions.
    
    Behavior is controlled by environment variables:
    - ALLOW_SELF_ACTIONS: Enable autonomous actions (default: False)
    - ALLOW_NETWORK: Enable network access (default: False)
    - AUTO_APPROVE: Auto-approve all actions (default: False, dangerous!)
    """
    
    def __init__(self):
        self.actions: List[AgentAction] = []
        self.allow_self_actions = os.getenv("ALLOW_SELF_ACTIONS", "false").lower() == "true"
        self.allow_network = os.getenv("ALLOW_NETWORK", "false").lower() == "true"
        self.auto_approve = os.getenv("AUTO_APPROVE", "false").lower() == "true"
        
        if self.auto_approve:
            logger.warning("AUTO_APPROVE is enabled - all actions will be auto-approved! Use with caution.")
        
    def request_approval(self, action: AgentAction) -> bool:
        """
        Request approval for an agent action.
        
        Returns True if approved, False otherwise.
        """
        self.actions.append(action)
        
        # Check environment flags
        if action.action_type == "network_request" and not self.allow_network:
            logger.info(f"Network action denied (ALLOW_NETWORK not set): {action.description}")
            action.approved = False
            return False
        
        if action.action_type in ["file_write", "code_execution"] and not self.allow_self_actions:
            logger.info(f"Self action denied (ALLOW_SELF_ACTIONS not set): {action.description}")
            action.approved = False
            return False
        
        # Auto-approve if enabled (dangerous!)
        if self.auto_approve:
            action.approved = True
            logger.info(f"Auto-approved: {action.action_type} - {action.description}")
            return True
        
        # Request manual approval
        print(f"\n🤖 Agent Action Approval Request")
        print(f"   Type: {action.action_type}")
        print(f"   Description: {action.description}")
        print(f"   Metadata: {action.metadata}")
        
        response = input("   Approve? (yes/no): ").strip().lower()
        action.approved = response in ["yes", "y"]
        
        if action.approved:
            logger.info(f"Approved: {action.action_type} - {action.description}")
        else:
            logger.info(f"Denied: {action.action_type} - {action.description}")
        
        return action.approved
    
    def record_result(self, action: AgentAction, result: Any = None, error: Optional[str] = None):
        """Record the result of an executed action."""
        action.result = result
        action.error = error
        logger.debug(f"Action result recorded: {action.action_type}")
    
    def get_audit_log(self) -> List[AgentAction]:
        """Return the full audit log."""
        return self.actions.copy()
    
    def print_audit_summary(self):
        """Print a summary of all actions."""
        print(f"\n📋 Agent Audit Summary")
        print(f"   Total actions: {len(self.actions)}")
        print(f"   Approved: {sum(1 for a in self.actions if a.approved)}")
        print(f"   Denied: {sum(1 for a in self.actions if not a.approved)}")
        print(f"   Errors: {sum(1 for a in self.actions if a.error)}")
        
        if self.actions:
            print(f"\n   Recent actions:")
            for action in self.actions[-5:]:
                status = "✓" if action.approved else "✗"
                print(f"     {status} [{action.timestamp.strftime('%H:%M:%S')}] {action.action_type}: {action.description}")

# Global auditor instance
_auditor = AgentAuditor()

def safe_execute(action_type: str, description: str, func: Callable, metadata: dict = None) -> Any:
    """
    Execute a function with approval gating and audit logging.
    
    Args:
        action_type: Type of action (e.g., "network_request")
        description: Human-readable description
        func: Function to execute if approved
        metadata: Additional metadata for auditing
        
    Returns:
        Result of func() if approved, None otherwise
    """
    action = AgentAction(
        action_type=action_type,
        description=description,
        metadata=metadata or {}
    )
    
    if not _auditor.request_approval(action):
        return None
    
    try:
        result = func()
        _auditor.record_result(action, result=result)
        return result
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        _auditor.record_result(action, error=error_msg)
        logger.error(f"Action failed: {error_msg}")
        raise

def get_auditor() -> AgentAuditor:
    """Get the global auditor instance."""
    return _auditor


# Example usage
if __name__ == "__main__":
    print("=== Agent Self-Auditing System Demo ===\n")
    
    # Example 1: Network request (requires approval)
    def make_request():
        return "Response from API"
    
    result = safe_execute(
        action_type="network_request",
        description="Fetch data from external API",
        func=make_request,
        metadata={"url": "https://api.example.com"}
    )
    
    if result:
        print(f"Result: {result}")
    
    # Example 2: File operation (requires approval)
    def write_file():
        return "File written successfully"
    
    result = safe_execute(
        action_type="file_write",
        description="Write configuration to disk",
        func=write_file,
        metadata={"path": "/tmp/config.json"}
    )
    
    # Print audit summary
    auditor = get_auditor()
    auditor.print_audit_summary()
    
    print("\n✓ Self-auditing system initialized")
    print("   Set ALLOW_SELF_ACTIONS=true to enable autonomous actions")
    print("   Set ALLOW_NETWORK=true to enable network access")
