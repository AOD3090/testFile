# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:srv/ServoOff.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ServoOff_Request(type):
    """Metaclass of message 'ServoOff_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'STOP_TYPE_QUICK_STO': 0,
        'STOP_TYPE_QUICK': 1,
        'STOP_TYPE_SLOW': 2,
        'STOP_TYPE_HOLD': 3,
        'STOP_TYPE_EMERGENCY': 3,
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
                'dsr_msgs2.srv.ServoOff_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__servo_off__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__servo_off__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__servo_off__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__servo_off__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__servo_off__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'STOP_TYPE_QUICK_STO': cls.__constants['STOP_TYPE_QUICK_STO'],
            'STOP_TYPE_QUICK': cls.__constants['STOP_TYPE_QUICK'],
            'STOP_TYPE_SLOW': cls.__constants['STOP_TYPE_SLOW'],
            'STOP_TYPE_HOLD': cls.__constants['STOP_TYPE_HOLD'],
            'STOP_TYPE_EMERGENCY': cls.__constants['STOP_TYPE_EMERGENCY'],
        }

    @property
    def STOP_TYPE_QUICK_STO(self):
        """Message constant 'STOP_TYPE_QUICK_STO'."""
        return Metaclass_ServoOff_Request.__constants['STOP_TYPE_QUICK_STO']

    @property
    def STOP_TYPE_QUICK(self):
        """Message constant 'STOP_TYPE_QUICK'."""
        return Metaclass_ServoOff_Request.__constants['STOP_TYPE_QUICK']

    @property
    def STOP_TYPE_SLOW(self):
        """Message constant 'STOP_TYPE_SLOW'."""
        return Metaclass_ServoOff_Request.__constants['STOP_TYPE_SLOW']

    @property
    def STOP_TYPE_HOLD(self):
        """Message constant 'STOP_TYPE_HOLD'."""
        return Metaclass_ServoOff_Request.__constants['STOP_TYPE_HOLD']

    @property
    def STOP_TYPE_EMERGENCY(self):
        """Message constant 'STOP_TYPE_EMERGENCY'."""
        return Metaclass_ServoOff_Request.__constants['STOP_TYPE_EMERGENCY']


class ServoOff_Request(metaclass=Metaclass_ServoOff_Request):
    """
    Message class 'ServoOff_Request'.

    Constants:
      STOP_TYPE_QUICK_STO
      STOP_TYPE_QUICK
      STOP_TYPE_SLOW
      STOP_TYPE_HOLD
      STOP_TYPE_EMERGENCY
    """

    __slots__ = [
        '_stop_type',
    ]

    _fields_and_field_types = {
        'stop_type': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.stop_type = kwargs.get('stop_type', int())

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
        if self.stop_type != other.stop_type:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def stop_type(self):
        """Message field 'stop_type'."""
        return self._stop_type

    @stop_type.setter
    def stop_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'stop_type' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'stop_type' field must be an integer in [-128, 127]"
        self._stop_type = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_ServoOff_Response(type):
    """Metaclass of message 'ServoOff_Response'."""

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
                'dsr_msgs2.srv.ServoOff_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__servo_off__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__servo_off__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__servo_off__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__servo_off__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__servo_off__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ServoOff_Response(metaclass=Metaclass_ServoOff_Response):
    """Message class 'ServoOff_Response'."""

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


class Metaclass_ServoOff(type):
    """Metaclass of service 'ServoOff'."""

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
                'dsr_msgs2.srv.ServoOff')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__servo_off

            from dsr_msgs2.srv import _servo_off
            if _servo_off.Metaclass_ServoOff_Request._TYPE_SUPPORT is None:
                _servo_off.Metaclass_ServoOff_Request.__import_type_support__()
            if _servo_off.Metaclass_ServoOff_Response._TYPE_SUPPORT is None:
                _servo_off.Metaclass_ServoOff_Response.__import_type_support__()


class ServoOff(metaclass=Metaclass_ServoOff):
    from dsr_msgs2.srv._servo_off import ServoOff_Request as Request
    from dsr_msgs2.srv._servo_off import ServoOff_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
