#
# rolx.sponsors.hollywood
# Formalize callbacks. Bring from deept and improve
#
#  - Events
#    EventType, + (context, e.g. "Enter" "Exit")
#    Context dictionary
#
#  - Callbacks
#    Container for all callbacks
#    Allow context managers (e.g. enter exit) to annotate events
#    Organize its callbacks to only call these that responds to an event
#    Manages execution order
#    Provide utility methods like "call_all_ignore_result"
#
#  - Callback
#      * callable with name and metadata
#      * __call__(event, context) => return protocol
#      * protocol to alter flow / terminate early?
#      * declare execution order preference (e.g. watch should be
#        "last" on enter and "first" on exit, to not measure also
#        other callbacks execution)
#
#  - Global registry of events (+ callbacks?) for documentation
#    and less error prone programming
#
