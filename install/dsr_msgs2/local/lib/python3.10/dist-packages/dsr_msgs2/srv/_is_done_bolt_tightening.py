# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:srv/IsDoneBoltTightening.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_IsDoneBoltTightening_Request(type):
    """Metaclass of message 'IsDoneBoltTightening_Request'."""

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
                'dsr_msgs2.srv.IsDoneBoltTightening_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__is_done_bolt_tightening__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__is_done_bolt_tightening__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__is_done_bolt_tightening__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__is_done_bolt_tightening__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__is_done_bolt_tightening__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class IsDoneBoltTightening_Request(metaclass=Metaclass_IsDoneBoltTightening_Request):
    """Message class 'IsDoneBoltTightening_Request'."""

    __slots__ = [
        '_m',
        '_timeout',
        '_axis',
    ]

    _fields_and_field_types = {
        'm': 'double',
        'timeout': 'double',
        'axis': 'int8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.m = kwargs.get('m', float())
        self.timeout = kwargs.get('timeout', float())
        self.axis = kwargs.get('axis', int())

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
        if self.m != other.m:
            return False
        if self.timeout != other.timeout:
            return False
        if self.axis != other.axis:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def m(self):
        """Message field 'm'."""
        return self._m

    @m.setter
    def m(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'm' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'm' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._m = value

    @builtins.property
    def timeout(self):
        """Message field 'timeout'."""
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'timeout' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'timeout' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._timeout = value

    @builtins.property
    def axis(self):
        """Message field 'axis'."""
        return self._axis

    @axis.setter
    def axis(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'axis' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'axis' field must be an integer in [-128, 127]"
        self._axis = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_IsDoneBoltTightening_Response(type):
    """Metaclass of message 'IsDoneBoltTightening_Response'."""

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
                'dsr_msgs2.srv.IsDoneBoltTightening_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__is_done_bolt_tightening__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__is_done_bolt_tightening__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__is_done_bolt_tightening__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__is_done_bolt_tightening__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__is_done_bolt_tightening__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class IsDoneBoltTightening_Response(metaclass=Metaclass_IsDoneBoltTightening_Response):
    """Message class 'IsDoneBoltTightening_Response'."""

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


class Metaclass_IsDoneBoltTightening(type):
    """Metaclass of service 'IsDoneBoltTightening'."""

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
                'dsr_msgs2.srv.IsDoneBoltTightening')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__is_done_bolt_tightening

            from dsr_msgs2.srv import _is_done_bolt_tightening
            if _is_done_bolt_tightening.Metaclass_IsDoneBoltTightening_Request._TYPE_SUPPORT is None:
                _is_done_bolt_tightening.Metaclass_IsDoneBoltTightening_Request.__import_type_support__()
            if _is_done_bolt_tightening.Metaclass_IsDoneBoltTightening_Response._TYPE_SUPPORT is None:
                _is_done_bolt_tightening.Metaclass_IsDoneBoltTightening_Response.__import_type_support__()


class IsDoneBoltTightening(metaclass=Metaclass_IsDoneBoltTightening):
    from dsr_msgs2.srv._is_done_bolt_tightening import IsDoneBoltTightening_Request as Request
    from dsr_msgs2.srv._is_done_bolt_tightening import IsDoneBoltTightening_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
