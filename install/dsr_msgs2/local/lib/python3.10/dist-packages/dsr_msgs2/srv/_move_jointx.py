# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:srv/MoveJointx.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'pos'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_MoveJointx_Request(type):
    """Metaclass of message 'MoveJointx_Request'."""

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
                'dsr_msgs2.srv.MoveJointx_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__move_jointx__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__move_jointx__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__move_jointx__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__move_jointx__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__move_jointx__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class MoveJointx_Request(metaclass=Metaclass_MoveJointx_Request):
    """Message class 'MoveJointx_Request'."""

    __slots__ = [
        '_pos',
        '_vel',
        '_acc',
        '_time',
        '_radius',
        '_ref',
        '_mode',
        '_blend_type',
        '_sol',
        '_sync_type',
    ]

    _fields_and_field_types = {
        'pos': 'double[6]',
        'vel': 'double',
        'acc': 'double',
        'time': 'double',
        'radius': 'double',
        'ref': 'int8',
        'mode': 'int8',
        'blend_type': 'int8',
        'sol': 'int8',
        'sync_type': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'pos' not in kwargs:
            self.pos = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.pos = kwargs.get('pos')
        self.vel = kwargs.get('vel', float())
        self.acc = kwargs.get('acc', float())
        self.time = kwargs.get('time', float())
        self.radius = kwargs.get('radius', float())
        self.ref = kwargs.get('ref', int())
        self.mode = kwargs.get('mode', int())
        self.blend_type = kwargs.get('blend_type', int())
        self.sol = kwargs.get('sol', int())
        self.sync_type = kwargs.get('sync_type', int())

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
        if any(self.pos != other.pos):
            return False
        if self.vel != other.vel:
            return False
        if self.acc != other.acc:
            return False
        if self.time != other.time:
            return False
        if self.radius != other.radius:
            return False
        if self.ref != other.ref:
            return False
        if self.mode != other.mode:
            return False
        if self.blend_type != other.blend_type:
            return False
        if self.sol != other.sol:
            return False
        if self.sync_type != other.sync_type:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def pos(self):
        """Message field 'pos'."""
        return self._pos

    @pos.setter
    def pos(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'pos' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'pos' numpy.ndarray() must have a size of 6"
            self._pos = value
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
                "The 'pos' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._pos = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def vel(self):
        """Message field 'vel'."""
        return self._vel

    @vel.setter
    def vel(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vel' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'vel' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._vel = value

    @builtins.property
    def acc(self):
        """Message field 'acc'."""
        return self._acc

    @acc.setter
    def acc(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'acc' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'acc' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._acc = value

    @builtins.property
    def time(self):
        """Message field 'time'."""
        return self._time

    @time.setter
    def time(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'time' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'time' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._time = value

    @builtins.property
    def radius(self):
        """Message field 'radius'."""
        return self._radius

    @radius.setter
    def radius(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'radius' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'radius' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._radius = value

    @builtins.property
    def ref(self):
        """Message field 'ref'."""
        return self._ref

    @ref.setter
    def ref(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'ref' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'ref' field must be an integer in [-128, 127]"
        self._ref = value

    @builtins.property
    def mode(self):
        """Message field 'mode'."""
        return self._mode

    @mode.setter
    def mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mode' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'mode' field must be an integer in [-128, 127]"
        self._mode = value

    @builtins.property
    def blend_type(self):
        """Message field 'blend_type'."""
        return self._blend_type

    @blend_type.setter
    def blend_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'blend_type' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'blend_type' field must be an integer in [-128, 127]"
        self._blend_type = value

    @builtins.property
    def sol(self):
        """Message field 'sol'."""
        return self._sol

    @sol.setter
    def sol(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'sol' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'sol' field must be an integer in [-128, 127]"
        self._sol = value

    @builtins.property
    def sync_type(self):
        """Message field 'sync_type'."""
        return self._sync_type

    @sync_type.setter
    def sync_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'sync_type' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'sync_type' field must be an integer in [-128, 127]"
        self._sync_type = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_MoveJointx_Response(type):
    """Metaclass of message 'MoveJointx_Response'."""

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
                'dsr_msgs2.srv.MoveJointx_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__move_jointx__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__move_jointx__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__move_jointx__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__move_jointx__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__move_jointx__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class MoveJointx_Response(metaclass=Metaclass_MoveJointx_Response):
    """Message class 'MoveJointx_Response'."""

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


class Metaclass_MoveJointx(type):
    """Metaclass of service 'MoveJointx'."""

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
                'dsr_msgs2.srv.MoveJointx')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__move_jointx

            from dsr_msgs2.srv import _move_jointx
            if _move_jointx.Metaclass_MoveJointx_Request._TYPE_SUPPORT is None:
                _move_jointx.Metaclass_MoveJointx_Request.__import_type_support__()
            if _move_jointx.Metaclass_MoveJointx_Response._TYPE_SUPPORT is None:
                _move_jointx.Metaclass_MoveJointx_Response.__import_type_support__()


class MoveJointx(metaclass=Metaclass_MoveJointx):
    from dsr_msgs2.srv._move_jointx import MoveJointx_Request as Request
    from dsr_msgs2.srv._move_jointx import MoveJointx_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
