"""Gateway-side accept policies, expressed over the shared decision space.

Policies here are stated as empirical rules over observable payment context:
amount, payee, resource, recent history. Nothing in this package is allowed to
read the scenario's ground truth, because a real gateway cannot.
"""
