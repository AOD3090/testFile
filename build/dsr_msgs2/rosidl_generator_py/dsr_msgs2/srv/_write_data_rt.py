# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:srv/WriteDataRt.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'external_force_torque'
# Member 'external_analog_input'
# Member 'external_analog_output'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_WriteDataRt_Request(type):
    """Metaclass of message 'WriteDataRt_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dsr_msgs2')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dsr_msgs2.srv.WriteDataRt_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__write_data_rt__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__write_data_rt__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__write_data_rt__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__write_data_rt__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__write_data_rt__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class WriteDataRt_Request(metaclass=Metaclass_WriteDataRt_Request):
    """Message class 'WriteDataRt_Request'."""

    __slots__ = [
        '_external_force_torque',
        '_external_digital_input',
        '_external_digital_output',
        '_external_analog_input',
        '_external_analog_output',
    ]

    _fields_and_field_types = {
        'external_force_torque': 'double[6]',
        'external_digital_input': 'int32',
        'external_digital_output': 'int32',
        'external_analog_input': 'double[6]',
        'external_analog_output': 'double[6]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'external_force_torque' not in kwargs:
            self.external_force_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.external_force_torque = kwargs.get('external_force_torque')
        self.external_digital_input = kwargs.get('external_digital_input', int())
        self.external_digital_output = kwargs.get('external_digital_output', int())
        if 'external_analog_input' not in kwargs:
            self.external_analog_input = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.external_analog_input = kwargs.get('external_analog_input')
        if 'external_analog_output' not in kwargs:
            self.external_analog_output = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.external_analog_output = kwargs.get('external_analog_output')

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if any(self.external_force_torque != other.external_force_torque):
            return False
        if self.external_digital_input != other.external_digital_input:
            return False
        if self.external_digital_output != other.external_digital_output:
            return False
        if any(self.external_analog_input != other.external_analog_input):
            return False
        if any(self.external_analog_output != other.external_analog_output):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def external_force_torque(self):
        """Message field 'external_force_torque'."""
        return self._external_force_torque

    @external_force_torque.setter
    def external_force_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'external_force_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'external_force_torque' numpy.ndarray() must have a size of 6"
            self._external_force_torque = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'external_force_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._external_force_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def external_digital_input(self):
        """Message field 'external_digital_input'."""
        return self._external_digital_input

    @external_digital_input.setter
    def external_digital_input(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'external_digital_input' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'external_digital_input' field must be an integer in [-2147483648, 2147483647]"
        self._external_digital_input = value

    @builtins.property
    def external_digital_output(self):
        """Message field 'external_digital_output'."""
        return self._external_digital_output

    @external_digital_output.setter
    def external_digital_output(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'external_digital_output' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'external_digital_output' field must be an integer in [-2147483648, 2147483647]"
        self._external_digital_output = value

    @builtins.property
    def external_analog_input(self):
        """Message field 'external_analog_input'."""
        return self._external_analog_input

    @external_analog_input.setter
    def external_analog_input(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'external_analog_input' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'external_analog_input' numpy.ndarray() must have a size of 6"
            self._external_analog_input = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'external_analog_input' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._external_analog_input = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def external_analog_output(self):
        """Message field 'external_analog_output'."""
        return self._external_analog_output

    @external_analog_output.setter
    def external_analog_output(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'external_analog_output' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'external_analog_output' numpy.ndarray() must have a size of 6"
            self._external_analog_output = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'external_analog_output' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._external_analog_output = numpy.array(value, dtype=numpy.float64)


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_WriteDataRt_Response(type):
    """Metaclass of message 'WriteDataRt_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dsr_msgs2')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dsr_msgs2.srv.WriteDataRt_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__write_data_rt__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__write_data_rt__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__write_data_rt__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__write_data_rt__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__write_data_rt__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class WriteDataRt_Response(metaclass=Metaclass_WriteDataRt_Response):
    """Message class 'WriteDataRt_Response'."""

    __slots__ = [
        '_success',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.success != other.success:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value


class Metaclass_WriteDataRt(type):
    """Metaclass of service 'WriteDataRt'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dsr_msgs2')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dsr_msgs2.srv.WriteDataRt')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__write_data_rt

            from dsr_msgs2.srv import _write_data_rt
            if _write_data_rt.Metaclass_WriteDataRt_Request._TYPE_SUPPORT is None:
                _write_data_rt.Metaclass_WriteDataRt_Request.__import_type_support__()
            if _write_data_rt.Metaclass_WriteDataRt_Response._TYPE_SUPPORT is None:
                _write_data_rt.Metaclass_WriteDataRt_Response.__import_type_support__()


class WriteDataRt(metaclass=Metaclass_WriteDataRt):
    from dsr_msgs2.srv._write_data_rt import WriteDataRt_Request as Request
    from dsr_msgs2.srv._write_data_rt import WriteDataRt_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
