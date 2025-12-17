# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:msg/RobotError.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RobotError(type):
    """Metaclass of message 'RobotError'."""

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
                'dsr_msgs2.msg.RobotError')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__robot_error
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__robot_error
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__robot_error
            cls._TYPE_SUPPORT = module.type_support_msg__msg__robot_error
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__robot_error

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RobotError(metaclass=Metaclass_RobotError):
    """Message class 'RobotError'."""

    __slots__ = [
        '_level',
        '_group',
        '_code',
        '_msg1',
        '_msg2',
        '_msg3',
    ]

    _fields_and_field_types = {
        'level': 'int32',
        'group': 'int32',
        'code': 'int32',
        'msg1': 'string',
        'msg2': 'string',
        'msg3': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.level = kwargs.get('level', int())
        self.group = kwargs.get('group', int())
        self.code = kwargs.get('code', int())
        self.msg1 = kwargs.get('msg1', str())
        self.msg2 = kwargs.get('msg2', str())
        self.msg3 = kwargs.get('msg3', str())

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
        if self.level != other.level:
            return False
        if self.group != other.group:
            return False
        if self.code != other.code:
            return False
        if self.msg1 != other.msg1:
            return False
        if self.msg2 != other.msg2:
            return False
        if self.msg3 != other.msg3:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def level(self):
        """Message field 'level'."""
        return self._level

    @level.setter
    def level(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'level' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'level' field must be an integer in [-2147483648, 2147483647]"
        self._level = value

    @builtins.property
    def group(self):
        """Message field 'group'."""
        return self._group

    @group.setter
    def group(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'group' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'group' field must be an integer in [-2147483648, 2147483647]"
        self._group = value

    @builtins.property
    def code(self):
        """Message field 'code'."""
        return self._code

    @code.setter
    def code(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'code' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'code' field must be an integer in [-2147483648, 2147483647]"
        self._code = value

    @builtins.property
    def msg1(self):
        """Message field 'msg1'."""
        return self._msg1

    @msg1.setter
    def msg1(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'msg1' field must be of type 'str'"
        self._msg1 = value

    @builtins.property
    def msg2(self):
        """Message field 'msg2'."""
        return self._msg2

    @msg2.setter
    def msg2(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'msg2' field must be of type 'str'"
        self._msg2 = value

    @builtins.property
    def msg3(self):
        """Message field 'msg3'."""
        return self._msg3

    @msg3.setter
    def msg3(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'msg3' field must be of type 'str'"
        self._msg3 = value
