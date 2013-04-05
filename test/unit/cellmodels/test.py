"""
"""

__author__ = "Marie E. Rognes (meg@simula.no), 2013"
__all__ = [""]

import unittest
from dolfin import *
from beatadjoint import supported_cell_models

class TestCellModels(unittest.TestCase):
    def test_create_cell_model(self):
        "Test that all supported cell models can be initialized and printed."
        for Model in supported_cell_models:
            model = Model()
            print "Successfully created %s." % model

    def test_create_cell_model_ics(self):
        "Test that all supported cell models have initial conditions."
        for Model in supported_cell_models:
            model = Model()
            print model.initial_conditions()

if __name__ == "__main__":
    print ""
    print "Testing cell models"
    print "-------------------"
    unittest.main()
