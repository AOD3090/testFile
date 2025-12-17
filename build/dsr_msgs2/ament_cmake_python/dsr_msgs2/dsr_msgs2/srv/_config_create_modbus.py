# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:srv/ConfigCreateModbus.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ConfigCreateModbus_Request(type):
    """Metaclass of message 'ConfigCreateModbus_Request'."""

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
                'dsr_msgs2.srv.ConfigCreateModbus_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__config_create_modbus__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__config_create_modbus__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__config_create_modbus__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__config_create_modbus__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__config_create_modbus__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ConfigCreateModbus_Request(metaclass=Metaclass_ConfigCreateModbus_Request):
    """Message class 'ConfigCreateModbus_Request'."""

    __slots__ = [
        '_name',
        '_ip',
        '_port',
        '_reg_type',
        '_index',
        '_value',
        '_slave_id',
    ]

    _fields_and_field_types = {
        'name': 'string',
        'ip': 'string',
        'port': 'int32',
        'reg_type': 'int8',
        'index': 'int8',
        'value': 'int8',
        'slave_id': 'int32',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.name = kwargs.get('name', str())
        self.ip = kwargs.get('ip', str())
        self.port = kwargs.get('port', int())
        self.reg_type = kwargs.get('reg_type', int())
        self.index = kwargs.get('index', int())
        self.value = kwargs.get('value', int())
        self.slave_id = kwargs.get('slave_id', int())

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
        if self.name != other.name:
            return False
        if self.ip != other.ip:
            return False
        if self.port != other.port:
            return False
        if self.reg_type != other.reg_type:
            return False
        if self.index != other.index:
            return False
        if self.value != other.value:
            return False
        if self.slave_id != other.slave_id:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def name(self):
        """Message field 'name'."""
        return self._name

    @name.setter
    def name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'name' field must be of type 'str'"
        self._name = value

    @builtins.property
    def ip(self):
        """Message field 'ip'."""
        return self._ip

    @ip.setter
    def ip(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'ip' field must be of type 'str'"
        self._ip = value

    @builtins.property
    def port(self):
        """Message field 'port'."""
        return self._port

    @port.setter
    def port(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'port' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'port' field must be an integer in [-2147483648, 2147483647]"
        self._port = value

    @builtins.property
    def reg_type(self):
        """Message field 'reg_type'."""
        return self._reg_type

    @reg_type.setter
    def reg_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'reg_type' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'reg_type' field must be an integer in [-128, 127]"
        self._reg_type = value

    @builtins.property
    def index(self):
        """Message field 'index'."""
        return self._index

    @index.setter
    def index(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'index' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'index' field must be an integer in [-128, 127]"
        self._index = value

    @builtins.property
    def value(self):
        """Message field 'value'."""
        return self._value

    @value.setter
    def value(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'value' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'value' field must be an integer in [-128, 127]"
        self._value = value

    @builtins.property
    def slave_id(self):
        """Message field 'slave_id'."""
        return self._slave_id

    @slave_id.setter
    def slave_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'slave_id' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'slave_id' field must be an integer in [-2147483648, 2147483647]"
        self._slave_id = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_ConfigCreateModbus_Response(type):
    """Metaclass of message 'ConfigCreateModbus_Response'."""

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
                'dsr_msgs2.srv.ConfigCreateModbus_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__config_create_modbus__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__config_create_modbus__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__config_create_modbus__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__config_create_modbus__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__config_create_modbus__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ConfigCreateModbus_Response(metaclass=Metaclass_ConfigCreateModbus_Response):
    """Message class 'ConfigCreateModbus_Response'."""

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


class Metaclass_ConfigCreateModbus(type):
    """Metaclass of service 'ConfigCreateModbus'."""

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
                'dsr_msgs2.srv.ConfigCreateModbus')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__config_create_modbus

            from dsr_msgs2.srv import _config_create_modbus
            if _config_create_modbus.Metaclass_ConfigCreateModbus_Request._TYPE_SUPPORT is None:
                _config_create_modbus.Metaclass_ConfigCreateModbus_Request.__import_type_support__()
            if _config_create_modbus.Metaclass_ConfigCreateModbus_Response._TYPE_SUPPORT is None:
                _config_create_modbus.Metaclass_ConfigCreateModbus_Response.__import_type_support__()


class ConfigCreateModbus(metaclass=Metaclass_ConfigCreateModbus):
    from dsr_msgs2.srv._config_create_modbus import ConfigCreateModbus_Request as Request
    from dsr_msgs2.srv._config_create_modbus import ConfigCreateModbus_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
