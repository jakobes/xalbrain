# Copyright (C) 2012 Johan Hake (hake.dev@gmail.com)
# Use and modify at will
# Last changed: 2012-10-17


__all__ = ["gotran2cellmodel"]

try:
    import gotran2

except Exception, e:
    print "Not possible to convert gotran model to cellmodel."
    raise e

# Gotran imports
from gotran2.codegeneration.codegenerator import CodeGenerator
from gotran2.common import error as gotran_error

_class_template = """
import ufl

from dolfin import *
from dolfin_adjoint import *

from beatadjoint.cellmodels import CardiacCellModel

class {ModelName}(CardiacCellModel):
    \"\"\"
{CLASSDOC}    
    \"\"\"
    def __init__(self, parameters=None):
        CardiacCellModel.__init__(self, parameters)

    def default_parameters(self):
        parameters = Parameters("{ModelName}")
{default_parameters}
        return parameters

    def I(self, v, s):
        \"\"\"
        Transmembrane current
        \"\"\"
{I_body}
        
        return current

    def F(self, v, s):
        \"\"\"
        Right hand side for ODE system
        \"\"\"
{F_body}
        return as_vector(F_expressions)

    def initial_conditions(self):
{initial_conditions}

        return ic

    def num_states(self):
        return {num_states}

    def __str__(self):
        return '{ModelName} cardiac cell model'
"""

_class_form = dict(
  ModelName="NOT_IMPLEMENTED",
  CLASSDOC="NOT_IMPLEMENTED",
  default_parameters="NOT_IMPLEMENTED",
  I_body="NOT_IMPLEMENTED",
  F_body="NOT_IMPLEMENTED",
  num_states="NOT_IMPLEMENTED",
  initial_conditions="NOT_IMPLEMENTED",
)

class CellModelGenerator(CodeGenerator):
    """
    Convert a Gotran model to a beat-adjoint compatible cell model
    """
    def __init__(self, oderepr):
        
        # Init base class
        super(CellModelGenerator, self).__init__(oderepr)

        # Get ode
        ode = oderepr.ode

        assert(not ode.is_dae)

        # Capitalize first letter of name
        name = oderepr.name
        self._name = name if name[0].isupper() else name[0].upper() + \
                     (name[1:] if len(name) > 1 else "")
        
        # Check we use correct ODERepresentation optimizations
        optimizations = [("use_state_names", True),
                         ("use_parameter_names", True),
                         ("keep_intermediates", False),
                         ("parameter_numerals", False),
                         ("use_cse", False)]

        for what, value in optimizations:
            if oderepr.optimization[what] != value:
                gotran_error("Got wrong optimization of ODERepresentation."\
                             " Expected {0} for '{1}' got {2}".format(\
                                 value, what, oderepr.optimization[what]))
        if ode.num_states < 2:
            gotran_error("expected the ODE to have more than 1 state")
            
        # Allowed names for the membrane potential
        membrane_pot = ["V", "v"]

        # Check that we only have one field state and that its name is v or V
        if not (ode.num_field_states == 1 and [state.name for state in \
                                    ode.iter_field_states()][0] in membrane_pot):
            gotran_error("expected ODE to have 1 and only 1 field state with "\
                         "name 'v' or 'V'")

        else:

            # The name of the membrane potential
            self.V_name = [state.name for state in ode.iter_field_states()][0]

        all_derivatives = [(derivatives[0], expr) for derivatives, expr \
                           in oderepr.iter_derivative_expr()]
    
        # Get the I and F expressions
        self._I_expression = [expr for derivatives, expr in all_derivatives \
                              if derivatives.name == self.V_name][0]

        # Get the used parameters
        I_used_parameters = set()
        for param in ode.iter_parameters():
            
            if param.sym in self._I_expression:
                I_used_parameters.add(param.name)
        self._I_used_parameters = list(I_used_parameters)
        
        self._F_expressions = [expr for derivatives, expr in all_derivatives \
                               if derivatives.name != self.V_name]

        # Get the used parameters
        F_used_parameters = set()
        for param in ode.iter_parameters():

            for expr in self._F_expressions:
                if param.sym in expr:
                    F_used_parameters.add(param.name)
        self._F_used_parameters = list(F_used_parameters)

        self.oderepr = oderepr

        # Create the class form and start fill it
        self._class_form = _class_form.copy()

        self._class_form["num_states"] = ode.num_states - 1
        self._class_form["ModelName"] = self.name
        self._class_form["default_parameters"] = self.default_parameters_body()
        self._class_form["F_body"] = self.F_body()
        self._class_form["I_body"] = self.I_body()
        self._class_form["initial_conditions"] = self.initial_conditions_body()
    @property
    def name(self):
        return self._name

    def generate(self):
        """
        Return a beat cell model file as a str
        """
        return _class_template.format(**self._class_form)
    
    def _common_body(self):
        """
        Return a common body for both I and F methods
        """
        ode = self.oderepr.ode

        body_lines = ["# Imports", "# No imports for now"]
        body_lines.append("")
        body_lines.append("# Assign states")
        if self.V_name != "v":
            body_lines.append("V = v")
        #body_lines.append("states = split(s)")
        body_lines.append("states = s")
        body_lines.append("assert(len(states) == {0})".format(ode.num_states-1))
        states = [state.name for state in ode.iter_states() \
                 if state.name != self.V_name]
        if len(states) == 1:
            body_lines.append("{0}, = states".format(states[0]))
        else:
            body_lines.append(", ".join(states) + " = states")
        body_lines.append("")
        
        return body_lines

    def I_body(self):
        """
        Generate code for the I body
        """

        from modelparameters.codegeneration import pythoncode

        # Get common body
        body_lines = self._common_body()

        body_lines.append("# Assign parameters")
        for param in self._I_used_parameters:
            body_lines.append("{0} = self._parameters["\
                              "\"{1}\"]".format(param, param))
        body_lines.append("")
        body_lines.append(pythoncode(self._I_expression, "current", namespace="ufl"))
        body_lines.append("")
        
        return "\n".join(self.indent_and_split_lines(body_lines, 2))

    def F_body(self):
        """
        Generate code for the F body
        """

        from modelparameters.codegeneration import pythoncode

        # Get common body
        body_lines = self._common_body()

        body_lines.append("# Assign parameters")
        for param in self._F_used_parameters:
            body_lines.append("{0} = self._parameters["\
                              "\"{1}\"]".format(param, param))
        body_lines.append("")
        body_lines.append("F_expressions = [\\")
        for expr in self._F_expressions:
            body_lines.append("")
            body_lines.append("    {0},".format(pythoncode(expr, namespace="ufl")))
        body_lines.append("    ]")
        body_lines.append("")
            
        return "\n".join(self.indent_and_split_lines(body_lines, 2))

    def default_parameters_body(self):
        """
        Generate code for the default parameter bod
        """

        ode = self.oderepr.ode
        body_lines = ["parameters.add(\"{0}\", {1})".format(param.name, param.init) \
                      for param in ode.iter_parameters()]
        return "\n".join(self.indent_and_split_lines(body_lines, 2))
        
    def initial_conditions_body(self):
        """
        Generate code for the ic body
        """

        ode = self.oderepr.ode
        
        # First get ic for v
        v_init = [state.init for state in ode.iter_states() if \
                  state.name == self.V_name][0][0]
        s_init, s_names = zip(*[(state.init, state.name) for state in ode.iter_states() \
                                if state.name != self.V_name])
        state_names = ", ".join("\"{0}\"".format(name) for name in s_names)
        state_init = ", ".join("{0} = {1}".format(name, value) for name, value in \
                               zip(s_names, s_init))
        body_lines = ["ic = Expression([\"{0}\", {1}],\\".format(\
            self.V_name, state_names)]
        body_lines.append("    {0}, {1})".format("{0}={1}".format(self.V_name, v_init), \
            state_init))

        return "\n".join(self.indent_and_split_lines(body_lines, 2))
