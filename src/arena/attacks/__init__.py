"""The attack catalog.

Each attack is a way to make a payment that is cryptographically valid and
outside what the delegator meant to authorize. Every attack here must ship
with a benign twin in `arena.benign`; an attack without one inflates the
block-rate of any policy that simply refuses more.
"""
