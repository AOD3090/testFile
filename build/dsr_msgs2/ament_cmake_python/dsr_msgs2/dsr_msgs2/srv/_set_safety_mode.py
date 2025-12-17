# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:srv/SetSafetyMode.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SetSafetyMode_Request(type):
    """Metaclass of message 'SetSafetyMode_Request'."""

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
                'dsr_msgs2.srv.SetSafetyMode_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_safety_mode__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_safety_mode__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_safety_mode__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_safety_mode__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_safety_mode__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetSafetyMode_Request(metaclass=Metaclass_SetSafetyMode_Request):
    """Message class 'SetSafetyMode_Request'."""

    __slots__ = [
        '_safety_mode',
        '_safety_event',
    ]

    _fields_and_field_types = {
        'safety_mode': 'int8',
        'safety_event': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.safety_mode = kwargs.get('safety_mode', int())
        self.safety_event = kwargs.get('safety_event', int())

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
        if self.safety_mode != other.safety_mode:
            return False
        if self.safety_event != other.safety_event:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def safety_mode(self):
        """Message field 'safety_mode'."""
        return self._safety_mode

    @safety_mode.setter
    def safety_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'safety_mode' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'safety_mode' field must be an integer in [-128, 127]"
        self._safety_mode = value

    @builtins.property
    def safety_event(self):
        """Message field 'safety_event'."""
        return self._safety_event

    @safety_event.setter
    def safety_event(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'safety_event' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'safety_event' field must be an integer in [-128, 127]"
        self._safety_event = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_SetSafetyMode_Response(type):
    """Metaclass of message 'SetSafetyMode_Response'."""

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
                'dsr_msgs2.srv.SetSafetyMode_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_safety_mode__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_safety_mode__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_safety_mode__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_safety_mode__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_safety_mode__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetSafetyMode_Response(metaclass=Metaclass_SetSafetyMode_Response):
    """Message class 'SetSafetyMode_Response'."""

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


class Metaclass_SetSafetyMode(type):
    """Metaclass of service 'SetSafetyMode'."""

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
                'dsr_msgs2.srv.SetSafetyMode')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__set_safety_mode

            from dsr_msgs2.srv import _set_safety_mode
            if _set_safety_mode.Metaclass_SetSafetyMode_Request._TYPE_SUPPORT is None:
                _set_safety_mode.Metaclass_SetSafetyMode_Request.__import_type_support__()
            if _set_safety_mode.Metaclass_SetSafetyMode_Response._TYPE_SUPPORT is None:
                _set_safety_mode.Metaclass_SetSafetyMode_Response.__import_type_support__()


class SetSafetyMode(metaclass=Metaclass_SetSafetyMode):
    from dsr_msgs2.srv._set_safety_mode import SetSafetyMode_Request as Request
    from dsr_msgs2.srv._set_safety_mode import SetSafetyMode_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
