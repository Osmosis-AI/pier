"""Pier's minimal mini-swe-agent runtime patch."""

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.exceptions import Submitted


class SubmittedResultInteractiveAgent(InteractiveAgent):
    """Record the successful submit action before mini exits."""

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        commands = [action["command"] for action in actions]
        outputs = []
        try:
            self._ask_confirmation_or_interrupt(commands)
            for action in actions:
                outputs.append(self.env.execute(action))
        except Submitted as exc:
            outputs.append({"output": "", "returncode": 0, "exception_info": ""})
            self._check_for_new_task_or_submit(exc)
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(
                    message, outputs, self.get_template_vars()
                )
            )
        return result
