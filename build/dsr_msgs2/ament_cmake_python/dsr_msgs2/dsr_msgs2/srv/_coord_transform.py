# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:srv/CoordTransform.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'pos_in'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CoordTransform_Request(type):
    """Metaclass of message 'CoordTransform_Request'."""

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
                'dsr_msgs2.srv.CoordTransform_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__coord_transform__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__coord_transform__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__coord_transform__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__coord_transform__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__coord_transform__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CoordTransform_Request(metaclass=Metaclass_CoordTransform_Request):
    """Message class 'CoordTransform_Request'."""

    __slots__ = [
        '_pos_in',
        '_ref_in',
        '_ref_out',
    ]

    _fields_and_field_types = {
        'pos_in': 'double[6]',
        'ref_in': 'int8',
        'ref_out': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'pos_in' not in kwargs:
            self.pos_in = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.pos_in = kwargs.get('pos_in')
        self.ref_in = kwargs.get('ref_in', int())
        self.ref_out = kwargs.get('ref_out', int())

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
        if any(self.pos_in != other.pos_in):
            return False
        if self.ref_in != other.ref_in:
            return False
        if self.ref_out != other.ref_out:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def pos_in(self):
        """Message field 'pos_in'."""
        return self._pos_in

    @pos_in.setter
    def pos_in(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'pos_in' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'pos_in' numpy.ndarray() must have a size of 6"
            self._pos_in = value
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
                "The 'pos_in' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._pos_in = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def ref_in(self):
        """Message field 'ref_in'."""
        return self._ref_in

    @ref_in.setter
    def ref_in(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'ref_in' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'ref_in' field must be an integer in [-128, 127]"
        self._ref_in = value

    @builtins.property
    def ref_out(self):
        """Message field 'ref_out'."""
        return self._ref_out

    @ref_out.setter
    def ref_out(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'ref_out' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'ref_out' field must be an integer in [-128, 127]"
        self._ref_out = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import math

# Member 'conv_posx'
# already imported above
# import numpy

# already imported above
# import rosidl_parser.definition


class Metaclass_CoordTransform_Response(type):
    """Metaclass of message 'CoordTransform_Response'."""

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
                'dsr_msgs2.srv.CoordTransform_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__coord_transform__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__coord_transform__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__coord_transform__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__coord_transform__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__coord_transform__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CoordTransform_Response(metaclass=Metaclass_CoordTransform_Response):
    """Message class 'CoordTransform_Response'."""

    __slots__ = [
        '_conv_posx',
        '_success',
    ]

    _fields_and_field_types = {
        'conv_posx': 'double[6]',
        'success': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'conv_posx' not in kwargs:
            self.conv_posx = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.conv_posx = kwargs.get('conv_posx')
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
        if any(self.conv_posx != other.conv_posx):
            return False
        if self.success != other.success:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def conv_posx(self):
        """Message field 'conv_posx'."""
        return self._conv_posx

    @conv_posx.setter
    def conv_posx(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'conv_posx' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'conv_posx' numpy.ndarray() must have a size of 6"
            self._conv_posx = value
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
                "The 'conv_posx' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._conv_posx = numpy.array(value, dtype=numpy.float64)

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


class Metaclass_CoordTransform(type):
    """Metaclass of service 'CoordTransform'."""

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
                'dsr_msgs2.srv.CoordTransform')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__coord_transform

            from dsr_msgs2.srv import _coord_transform
            if _coord_transform.Metaclass_CoordTransform_Request._TYPE_SUPPORT is None:
                _coord_transform.Metaclass_CoordTransform_Request.__import_type_support__()
            if _coord_transform.Metaclass_CoordTransform_Response._TYPE_SUPPORT is None:
                _coord_transform.Metaclass_CoordTransform_Response.__import_type_support__()


class CoordTransform(metaclass=Metaclass_CoordTransform):
    from dsr_msgs2.srv._coord_transform import CoordTransform_Request as Request
    from dsr_msgs2.srv._coord_transform import CoordTransform_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
