# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:msg/RobotStateRt.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'actual_joint_position'
# Member 'actual_joint_position_abs'
# Member 'actual_joint_velocity'
# Member 'actual_joint_velocity_abs'
# Member 'actual_tcp_position'
# Member 'actual_tcp_velocity'
# Member 'actual_flange_position'
# Member 'actual_flange_velocity'
# Member 'actual_motor_torque'
# Member 'actual_joint_torque'
# Member 'raw_joint_torque'
# Member 'raw_force_torque'
# Member 'external_joint_torque'
# Member 'external_tcp_force'
# Member 'target_joint_position'
# Member 'target_joint_velocity'
# Member 'target_joint_acceleration'
# Member 'target_motor_torque'
# Member 'target_tcp_position'
# Member 'target_tcp_velocity'
# Member 'gravity_torque'
# Member 'joint_temperature'
# Member 'controller_analog_input_type'
# Member 'controller_analog_input'
# Member 'controller_analog_output_type'
# Member 'controller_analog_output'
# Member 'flange_analog_input'
# Member 'external_encoder_strobe_count'
# Member 'external_encoder_count'
# Member 'goal_joint_position'
# Member 'goal_tcp_position'
# Member 'reserved'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RobotStateRt(type):
    """Metaclass of message 'RobotStateRt'."""

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
                'dsr_msgs2.msg.RobotStateRt')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__robot_state_rt
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__robot_state_rt
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__robot_state_rt
            cls._TYPE_SUPPORT = module.type_support_msg__msg__robot_state_rt
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__robot_state_rt

            from std_msgs.msg import Float64MultiArray
            if Float64MultiArray.__class__._TYPE_SUPPORT is None:
                Float64MultiArray.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RobotStateRt(metaclass=Metaclass_RobotStateRt):
    """Message class 'RobotStateRt'."""

    __slots__ = [
        '_time_stamp',
        '_actual_joint_position',
        '_actual_joint_position_abs',
        '_actual_joint_velocity',
        '_actual_joint_velocity_abs',
        '_actual_tcp_position',
        '_actual_tcp_velocity',
        '_actual_flange_position',
        '_actual_flange_velocity',
        '_actual_motor_torque',
        '_actual_joint_torque',
        '_raw_joint_torque',
        '_raw_force_torque',
        '_external_joint_torque',
        '_external_tcp_force',
        '_target_joint_position',
        '_target_joint_velocity',
        '_target_joint_acceleration',
        '_target_motor_torque',
        '_target_tcp_position',
        '_target_tcp_velocity',
        '_jacobian_matrix',
        '_gravity_torque',
        '_coriolis_matrix',
        '_mass_matrix',
        '_solution_space',
        '_singularity',
        '_operation_speed_rate',
        '_joint_temperature',
        '_controller_digital_input',
        '_controller_digital_output',
        '_controller_analog_input_type',
        '_controller_analog_input',
        '_controller_analog_output_type',
        '_controller_analog_output',
        '_flange_digital_input',
        '_flange_digital_output',
        '_flange_analog_input',
        '_external_encoder_strobe_count',
        '_external_encoder_count',
        '_goal_joint_position',
        '_goal_tcp_position',
        '_robot_mode',
        '_robot_state',
        '_control_mode',
        '_reserved',
    ]

    _fields_and_field_types = {
        'time_stamp': 'double',
        'actual_joint_position': 'double[6]',
        'actual_joint_position_abs': 'double[6]',
        'actual_joint_velocity': 'double[6]',
        'actual_joint_velocity_abs': 'double[6]',
        'actual_tcp_position': 'double[6]',
        'actual_tcp_velocity': 'double[6]',
        'actual_flange_position': 'double[6]',
        'actual_flange_velocity': 'double[6]',
        'actual_motor_torque': 'double[6]',
        'actual_joint_torque': 'double[6]',
        'raw_joint_torque': 'double[6]',
        'raw_force_torque': 'double[6]',
        'external_joint_torque': 'double[6]',
        'external_tcp_force': 'double[6]',
        'target_joint_position': 'double[6]',
        'target_joint_velocity': 'double[6]',
        'target_joint_acceleration': 'double[6]',
        'target_motor_torque': 'double[6]',
        'target_tcp_position': 'double[6]',
        'target_tcp_velocity': 'double[6]',
        'jacobian_matrix': 'sequence<std_msgs/Float64MultiArray>',
        'gravity_torque': 'double[6]',
        'coriolis_matrix': 'sequence<std_msgs/Float64MultiArray>',
        'mass_matrix': 'sequence<std_msgs/Float64MultiArray>',
        'solution_space': 'uint16',
        'singularity': 'double',
        'operation_speed_rate': 'double',
        'joint_temperature': 'double[6]',
        'controller_digital_input': 'uint16',
        'controller_digital_output': 'uint16',
        'controller_analog_input_type': 'uint8[2]',
        'controller_analog_input': 'double[2]',
        'controller_analog_output_type': 'uint8[2]',
        'controller_analog_output': 'double[2]',
        'flange_digital_input': 'uint8',
        'flange_digital_output': 'uint8',
        'flange_analog_input': 'double[4]',
        'external_encoder_strobe_count': 'uint8[2]',
        'external_encoder_count': 'uint16[2]',
        'goal_joint_position': 'double[6]',
        'goal_tcp_position': 'double[6]',
        'robot_mode': 'uint8',
        'robot_state': 'uint8',
        'control_mode': 'uint16',
        'reserved': 'uint8[256]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray')),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray')),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 2),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint16'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 256),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.time_stamp = kwargs.get('time_stamp', float())
        if 'actual_joint_position' not in kwargs:
            self.actual_joint_position = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_joint_position = kwargs.get('actual_joint_position')
        if 'actual_joint_position_abs' not in kwargs:
            self.actual_joint_position_abs = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_joint_position_abs = kwargs.get('actual_joint_position_abs')
        if 'actual_joint_velocity' not in kwargs:
            self.actual_joint_velocity = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_joint_velocity = kwargs.get('actual_joint_velocity')
        if 'actual_joint_velocity_abs' not in kwargs:
            self.actual_joint_velocity_abs = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_joint_velocity_abs = kwargs.get('actual_joint_velocity_abs')
        if 'actual_tcp_position' not in kwargs:
            self.actual_tcp_position = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_tcp_position = kwargs.get('actual_tcp_position')
        if 'actual_tcp_velocity' not in kwargs:
            self.actual_tcp_velocity = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_tcp_velocity = kwargs.get('actual_tcp_velocity')
        if 'actual_flange_position' not in kwargs:
            self.actual_flange_position = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_flange_position = kwargs.get('actual_flange_position')
        if 'actual_flange_velocity' not in kwargs:
            self.actual_flange_velocity = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_flange_velocity = kwargs.get('actual_flange_velocity')
        if 'actual_motor_torque' not in kwargs:
            self.actual_motor_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_motor_torque = kwargs.get('actual_motor_torque')
        if 'actual_joint_torque' not in kwargs:
            self.actual_joint_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_joint_torque = kwargs.get('actual_joint_torque')
        if 'raw_joint_torque' not in kwargs:
            self.raw_joint_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.raw_joint_torque = kwargs.get('raw_joint_torque')
        if 'raw_force_torque' not in kwargs:
            self.raw_force_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.raw_force_torque = kwargs.get('raw_force_torque')
        if 'external_joint_torque' not in kwargs:
            self.external_joint_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.external_joint_torque = kwargs.get('external_joint_torque')
        if 'external_tcp_force' not in kwargs:
            self.external_tcp_force = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.external_tcp_force = kwargs.get('external_tcp_force')
        if 'target_joint_position' not in kwargs:
            self.target_joint_position = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_joint_position = kwargs.get('target_joint_position')
        if 'target_joint_velocity' not in kwargs:
            self.target_joint_velocity = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_joint_velocity = kwargs.get('target_joint_velocity')
        if 'target_joint_acceleration' not in kwargs:
            self.target_joint_acceleration = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_joint_acceleration = kwargs.get('target_joint_acceleration')
        if 'target_motor_torque' not in kwargs:
            self.target_motor_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_motor_torque = kwargs.get('target_motor_torque')
        if 'target_tcp_position' not in kwargs:
            self.target_tcp_position = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_tcp_position = kwargs.get('target_tcp_position')
        if 'target_tcp_velocity' not in kwargs:
            self.target_tcp_velocity = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_tcp_velocity = kwargs.get('target_tcp_velocity')
        self.jacobian_matrix = kwargs.get('jacobian_matrix', [])
        if 'gravity_torque' not in kwargs:
            self.gravity_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.gravity_torque = kwargs.get('gravity_torque')
        self.coriolis_matrix = kwargs.get('coriolis_matrix', [])
        self.mass_matrix = kwargs.get('mass_matrix', [])
        self.solution_space = kwargs.get('solution_space', int())
        self.singularity = kwargs.get('singularity', float())
        self.operation_speed_rate = kwargs.get('operation_speed_rate', float())
        if 'joint_temperature' not in kwargs:
            self.joint_temperature = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.joint_temperature = kwargs.get('joint_temperature')
        self.controller_digital_input = kwargs.get('controller_digital_input', int())
        self.controller_digital_output = kwargs.get('controller_digital_output', int())
        if 'controller_analog_input_type' not in kwargs:
            self.controller_analog_input_type = numpy.zeros(2, dtype=numpy.uint8)
        else:
            self.controller_analog_input_type = kwargs.get('controller_analog_input_type')
        if 'controller_analog_input' not in kwargs:
            self.controller_analog_input = numpy.zeros(2, dtype=numpy.float64)
        else:
            self.controller_analog_input = kwargs.get('controller_analog_input')
        if 'controller_analog_output_type' not in kwargs:
            self.controller_analog_output_type = numpy.zeros(2, dtype=numpy.uint8)
        else:
            self.controller_analog_output_type = kwargs.get('controller_analog_output_type')
        if 'controller_analog_output' not in kwargs:
            self.controller_analog_output = numpy.zeros(2, dtype=numpy.float64)
        else:
            self.controller_analog_output = kwargs.get('controller_analog_output')
        self.flange_digital_input = kwargs.get('flange_digital_input', int())
        self.flange_digital_output = kwargs.get('flange_digital_output', int())
        if 'flange_analog_input' not in kwargs:
            self.flange_analog_input = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.flange_analog_input = kwargs.get('flange_analog_input')
        if 'external_encoder_strobe_count' not in kwargs:
            self.external_encoder_strobe_count = numpy.zeros(2, dtype=numpy.uint8)
        else:
            self.external_encoder_strobe_count = kwargs.get('external_encoder_strobe_count')
        if 'external_encoder_count' not in kwargs:
            self.external_encoder_count = numpy.zeros(2, dtype=numpy.uint16)
        else:
            self.external_encoder_count = kwargs.get('external_encoder_count')
        if 'goal_joint_position' not in kwargs:
            self.goal_joint_position = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.goal_joint_position = kwargs.get('goal_joint_position')
        if 'goal_tcp_position' not in kwargs:
            self.goal_tcp_position = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.goal_tcp_position = kwargs.get('goal_tcp_position')
        self.robot_mode = kwargs.get('robot_mode', int())
        self.robot_state = kwargs.get('robot_state', int())
        self.control_mode = kwargs.get('control_mode', int())
        if 'reserved' not in kwargs:
            self.reserved = numpy.zeros(256, dtype=numpy.uint8)
        else:
            self.reserved = kwargs.get('reserved')

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
        if self.time_stamp != other.time_stamp:
            return False
        if any(self.actual_joint_position != other.actual_joint_position):
            return False
        if any(self.actual_joint_position_abs != other.actual_joint_position_abs):
            return False
        if any(self.actual_joint_velocity != other.actual_joint_velocity):
            return False
        if any(self.actual_joint_velocity_abs != other.actual_joint_velocity_abs):
            return False
        if any(self.actual_tcp_position != other.actual_tcp_position):
            return False
        if any(self.actual_tcp_velocity != other.actual_tcp_velocity):
            return False
        if any(self.actual_flange_position != other.actual_flange_position):
            return False
        if any(self.actual_flange_velocity != other.actual_flange_velocity):
            return False
        if any(self.actual_motor_torque != other.actual_motor_torque):
            return False
        if any(self.actual_joint_torque != other.actual_joint_torque):
            return False
        if any(self.raw_joint_torque != other.raw_joint_torque):
            return False
        if any(self.raw_force_torque != other.raw_force_torque):
            return False
        if any(self.external_joint_torque != other.external_joint_torque):
            return False
        if any(self.external_tcp_force != other.external_tcp_force):
            return False
        if any(self.target_joint_position != other.target_joint_position):
            return False
        if any(self.target_joint_velocity != other.target_joint_velocity):
            return False
        if any(self.target_joint_acceleration != other.target_joint_acceleration):
            return False
        if any(self.target_motor_torque != other.target_motor_torque):
            return False
        if any(self.target_tcp_position != other.target_tcp_position):
            return False
        if any(self.target_tcp_velocity != other.target_tcp_velocity):
            return False
        if self.jacobian_matrix != other.jacobian_matrix:
            return False
        if any(self.gravity_torque != other.gravity_torque):
            return False
        if self.coriolis_matrix != other.coriolis_matrix:
            return False
        if self.mass_matrix != other.mass_matrix:
            return False
        if self.solution_space != other.solution_space:
            return False
        if self.singularity != other.singularity:
            return False
        if self.operation_speed_rate != other.operation_speed_rate:
            return False
        if any(self.joint_temperature != other.joint_temperature):
            return False
        if self.controller_digital_input != other.controller_digital_input:
            return False
        if self.controller_digital_output != other.controller_digital_output:
            return False
        if any(self.controller_analog_input_type != other.controller_analog_input_type):
            return False
        if any(self.controller_analog_input != other.controller_analog_input):
            return False
        if any(self.controller_analog_output_type != other.controller_analog_output_type):
            return False
        if any(self.controller_analog_output != other.controller_analog_output):
            return False
        if self.flange_digital_input != other.flange_digital_input:
            return False
        if self.flange_digital_output != other.flange_digital_output:
            return False
        if any(self.flange_analog_input != other.flange_analog_input):
            return False
        if any(self.external_encoder_strobe_count != other.external_encoder_strobe_count):
            return False
        if any(self.external_encoder_count != other.external_encoder_count):
            return False
        if any(self.goal_joint_position != other.goal_joint_position):
            return False
        if any(self.goal_tcp_position != other.goal_tcp_position):
            return False
        if self.robot_mode != other.robot_mode:
            return False
        if self.robot_state != other.robot_state:
            return False
        if self.control_mode != other.control_mode:
            return False
        if any(self.reserved != other.reserved):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def time_stamp(self):
        """Message field 'time_stamp'."""
        return self._time_stamp

    @time_stamp.setter
    def time_stamp(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'time_stamp' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'time_stamp' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._time_stamp = value

    @builtins.property
    def actual_joint_position(self):
        """Message field 'actual_joint_position'."""
        return self._actual_joint_position

    @actual_joint_position.setter
    def actual_joint_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_joint_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_joint_position' numpy.ndarray() must have a size of 6"
            self._actual_joint_position = value
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
                "The 'actual_joint_position' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_joint_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_joint_position_abs(self):
        """Message field 'actual_joint_position_abs'."""
        return self._actual_joint_position_abs

    @actual_joint_position_abs.setter
    def actual_joint_position_abs(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_joint_position_abs' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_joint_position_abs' numpy.ndarray() must have a size of 6"
            self._actual_joint_position_abs = value
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
                "The 'actual_joint_position_abs' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_joint_position_abs = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_joint_velocity(self):
        """Message field 'actual_joint_velocity'."""
        return self._actual_joint_velocity

    @actual_joint_velocity.setter
    def actual_joint_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_joint_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_joint_velocity' numpy.ndarray() must have a size of 6"
            self._actual_joint_velocity = value
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
                "The 'actual_joint_velocity' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_joint_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_joint_velocity_abs(self):
        """Message field 'actual_joint_velocity_abs'."""
        return self._actual_joint_velocity_abs

    @actual_joint_velocity_abs.setter
    def actual_joint_velocity_abs(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_joint_velocity_abs' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_joint_velocity_abs' numpy.ndarray() must have a size of 6"
            self._actual_joint_velocity_abs = value
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
                "The 'actual_joint_velocity_abs' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_joint_velocity_abs = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_tcp_position(self):
        """Message field 'actual_tcp_position'."""
        return self._actual_tcp_position

    @actual_tcp_position.setter
    def actual_tcp_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_tcp_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_tcp_position' numpy.ndarray() must have a size of 6"
            self._actual_tcp_position = value
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
                "The 'actual_tcp_position' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_tcp_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_tcp_velocity(self):
        """Message field 'actual_tcp_velocity'."""
        return self._actual_tcp_velocity

    @actual_tcp_velocity.setter
    def actual_tcp_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_tcp_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_tcp_velocity' numpy.ndarray() must have a size of 6"
            self._actual_tcp_velocity = value
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
                "The 'actual_tcp_velocity' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_tcp_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_flange_position(self):
        """Message field 'actual_flange_position'."""
        return self._actual_flange_position

    @actual_flange_position.setter
    def actual_flange_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_flange_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_flange_position' numpy.ndarray() must have a size of 6"
            self._actual_flange_position = value
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
                "The 'actual_flange_position' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_flange_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_flange_velocity(self):
        """Message field 'actual_flange_velocity'."""
        return self._actual_flange_velocity

    @actual_flange_velocity.setter
    def actual_flange_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_flange_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_flange_velocity' numpy.ndarray() must have a size of 6"
            self._actual_flange_velocity = value
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
                "The 'actual_flange_velocity' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_flange_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_motor_torque(self):
        """Message field 'actual_motor_torque'."""
        return self._actual_motor_torque

    @actual_motor_torque.setter
    def actual_motor_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_motor_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_motor_torque' numpy.ndarray() must have a size of 6"
            self._actual_motor_torque = value
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
                "The 'actual_motor_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_motor_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_joint_torque(self):
        """Message field 'actual_joint_torque'."""
        return self._actual_joint_torque

    @actual_joint_torque.setter
    def actual_joint_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_joint_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_joint_torque' numpy.ndarray() must have a size of 6"
            self._actual_joint_torque = value
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
                "The 'actual_joint_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_joint_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def raw_joint_torque(self):
        """Message field 'raw_joint_torque'."""
        return self._raw_joint_torque

    @raw_joint_torque.setter
    def raw_joint_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'raw_joint_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'raw_joint_torque' numpy.ndarray() must have a size of 6"
            self._raw_joint_torque = value
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
                "The 'raw_joint_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._raw_joint_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def raw_force_torque(self):
        """Message field 'raw_force_torque'."""
        return self._raw_force_torque

    @raw_force_torque.setter
    def raw_force_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'raw_force_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'raw_force_torque' numpy.ndarray() must have a size of 6"
            self._raw_force_torque = value
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
                "The 'raw_force_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._raw_force_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def external_joint_torque(self):
        """Message field 'external_joint_torque'."""
        return self._external_joint_torque

    @external_joint_torque.setter
    def external_joint_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'external_joint_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'external_joint_torque' numpy.ndarray() must have a size of 6"
            self._external_joint_torque = value
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
                "The 'external_joint_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._external_joint_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def external_tcp_force(self):
        """Message field 'external_tcp_force'."""
        return self._external_tcp_force

    @external_tcp_force.setter
    def external_tcp_force(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'external_tcp_force' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'external_tcp_force' numpy.ndarray() must have a size of 6"
            self._external_tcp_force = value
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
                "The 'external_tcp_force' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._external_tcp_force = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_joint_position(self):
        """Message field 'target_joint_position'."""
        return self._target_joint_position

    @target_joint_position.setter
    def target_joint_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_joint_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_joint_position' numpy.ndarray() must have a size of 6"
            self._target_joint_position = value
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
                "The 'target_joint_position' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_joint_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_joint_velocity(self):
        """Message field 'target_joint_velocity'."""
        return self._target_joint_velocity

    @target_joint_velocity.setter
    def target_joint_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_joint_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_joint_velocity' numpy.ndarray() must have a size of 6"
            self._target_joint_velocity = value
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
                "The 'target_joint_velocity' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_joint_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_joint_acceleration(self):
        """Message field 'target_joint_acceleration'."""
        return self._target_joint_acceleration

    @target_joint_acceleration.setter
    def target_joint_acceleration(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_joint_acceleration' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_joint_acceleration' numpy.ndarray() must have a size of 6"
            self._target_joint_acceleration = value
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
                "The 'target_joint_acceleration' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_joint_acceleration = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_motor_torque(self):
        """Message field 'target_motor_torque'."""
        return self._target_motor_torque

    @target_motor_torque.setter
    def target_motor_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_motor_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_motor_torque' numpy.ndarray() must have a size of 6"
            self._target_motor_torque = value
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
                "The 'target_motor_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_motor_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_tcp_position(self):
        """Message field 'target_tcp_position'."""
        return self._target_tcp_position

    @target_tcp_position.setter
    def target_tcp_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_tcp_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_tcp_position' numpy.ndarray() must have a size of 6"
            self._target_tcp_position = value
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
                "The 'target_tcp_position' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_tcp_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_tcp_velocity(self):
        """Message field 'target_tcp_velocity'."""
        return self._target_tcp_velocity

    @target_tcp_velocity.setter
    def target_tcp_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_tcp_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_tcp_velocity' numpy.ndarray() must have a size of 6"
            self._target_tcp_velocity = value
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
                "The 'target_tcp_velocity' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_tcp_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def jacobian_matrix(self):
        """Message field 'jacobian_matrix'."""
        return self._jacobian_matrix

    @jacobian_matrix.setter
    def jacobian_matrix(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
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
                 all(isinstance(v, Float64MultiArray) for v in value) and
                 True), \
                "The 'jacobian_matrix' field must be a set or sequence and each value of type 'Float64MultiArray'"
        self._jacobian_matrix = value

    @builtins.property
    def gravity_torque(self):
        """Message field 'gravity_torque'."""
        return self._gravity_torque

    @gravity_torque.setter
    def gravity_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'gravity_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'gravity_torque' numpy.ndarray() must have a size of 6"
            self._gravity_torque = value
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
                "The 'gravity_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._gravity_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def coriolis_matrix(self):
        """Message field 'coriolis_matrix'."""
        return self._coriolis_matrix

    @coriolis_matrix.setter
    def coriolis_matrix(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
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
                 all(isinstance(v, Float64MultiArray) for v in value) and
                 True), \
                "The 'coriolis_matrix' field must be a set or sequence and each value of type 'Float64MultiArray'"
        self._coriolis_matrix = value

    @builtins.property
    def mass_matrix(self):
        """Message field 'mass_matrix'."""
        return self._mass_matrix

    @mass_matrix.setter
    def mass_matrix(self, value):
        if __debug__:
            from std_msgs.msg import Float64MultiArray
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
                 all(isinstance(v, Float64MultiArray) for v in value) and
                 True), \
                "The 'mass_matrix' field must be a set or sequence and each value of type 'Float64MultiArray'"
        self._mass_matrix = value

    @builtins.property
    def solution_space(self):
        """Message field 'solution_space'."""
        return self._solution_space

    @solution_space.setter
    def solution_space(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'solution_space' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'solution_space' field must be an unsigned integer in [0, 65535]"
        self._solution_space = value

    @builtins.property
    def singularity(self):
        """Message field 'singularity'."""
        return self._singularity

    @singularity.setter
    def singularity(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'singularity' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'singularity' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._singularity = value

    @builtins.property
    def operation_speed_rate(self):
        """Message field 'operation_speed_rate'."""
        return self._operation_speed_rate

    @operation_speed_rate.setter
    def operation_speed_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'operation_speed_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'operation_speed_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._operation_speed_rate = value

    @builtins.property
    def joint_temperature(self):
        """Message field 'joint_temperature'."""
        return self._joint_temperature

    @joint_temperature.setter
    def joint_temperature(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'joint_temperature' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'joint_temperature' numpy.ndarray() must have a size of 6"
            self._joint_temperature = value
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
                "The 'joint_temperature' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joint_temperature = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def controller_digital_input(self):
        """Message field 'controller_digital_input'."""
        return self._controller_digital_input

    @controller_digital_input.setter
    def controller_digital_input(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'controller_digital_input' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'controller_digital_input' field must be an unsigned integer in [0, 65535]"
        self._controller_digital_input = value

    @builtins.property
    def controller_digital_output(self):
        """Message field 'controller_digital_output'."""
        return self._controller_digital_output

    @controller_digital_output.setter
    def controller_digital_output(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'controller_digital_output' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'controller_digital_output' field must be an unsigned integer in [0, 65535]"
        self._controller_digital_output = value

    @builtins.property
    def controller_analog_input_type(self):
        """Message field 'controller_analog_input_type'."""
        return self._controller_analog_input_type

    @controller_analog_input_type.setter
    def controller_analog_input_type(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'controller_analog_input_type' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 2, \
                "The 'controller_analog_input_type' numpy.ndarray() must have a size of 2"
            self._controller_analog_input_type = value
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
                 len(value) == 2 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'controller_analog_input_type' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._controller_analog_input_type = numpy.array(value, dtype=numpy.uint8)

    @builtins.property
    def controller_analog_input(self):
        """Message field 'controller_analog_input'."""
        return self._controller_analog_input

    @controller_analog_input.setter
    def controller_analog_input(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'controller_analog_input' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 2, \
                "The 'controller_analog_input' numpy.ndarray() must have a size of 2"
            self._controller_analog_input = value
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
                 len(value) == 2 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'controller_analog_input' field must be a set or sequence with length 2 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._controller_analog_input = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def controller_analog_output_type(self):
        """Message field 'controller_analog_output_type'."""
        return self._controller_analog_output_type

    @controller_analog_output_type.setter
    def controller_analog_output_type(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'controller_analog_output_type' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 2, \
                "The 'controller_analog_output_type' numpy.ndarray() must have a size of 2"
            self._controller_analog_output_type = value
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
                 len(value) == 2 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'controller_analog_output_type' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._controller_analog_output_type = numpy.array(value, dtype=numpy.uint8)

    @builtins.property
    def controller_analog_output(self):
        """Message field 'controller_analog_output'."""
        return self._controller_analog_output

    @controller_analog_output.setter
    def controller_analog_output(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'controller_analog_output' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 2, \
                "The 'controller_analog_output' numpy.ndarray() must have a size of 2"
            self._controller_analog_output = value
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
                 len(value) == 2 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'controller_analog_output' field must be a set or sequence with length 2 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._controller_analog_output = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def flange_digital_input(self):
        """Message field 'flange_digital_input'."""
        return self._flange_digital_input

    @flange_digital_input.setter
    def flange_digital_input(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'flange_digital_input' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'flange_digital_input' field must be an unsigned integer in [0, 255]"
        self._flange_digital_input = value

    @builtins.property
    def flange_digital_output(self):
        """Message field 'flange_digital_output'."""
        return self._flange_digital_output

    @flange_digital_output.setter
    def flange_digital_output(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'flange_digital_output' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'flange_digital_output' field must be an unsigned integer in [0, 255]"
        self._flange_digital_output = value

    @builtins.property
    def flange_analog_input(self):
        """Message field 'flange_analog_input'."""
        return self._flange_analog_input

    @flange_analog_input.setter
    def flange_analog_input(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'flange_analog_input' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'flange_analog_input' numpy.ndarray() must have a size of 4"
            self._flange_analog_input = value
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
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'flange_analog_input' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._flange_analog_input = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def external_encoder_strobe_count(self):
        """Message field 'external_encoder_strobe_count'."""
        return self._external_encoder_strobe_count

    @external_encoder_strobe_count.setter
    def external_encoder_strobe_count(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'external_encoder_strobe_count' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 2, \
                "The 'external_encoder_strobe_count' numpy.ndarray() must have a size of 2"
            self._external_encoder_strobe_count = value
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
                 len(value) == 2 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'external_encoder_strobe_count' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._external_encoder_strobe_count = numpy.array(value, dtype=numpy.uint8)

    @builtins.property
    def external_encoder_count(self):
        """Message field 'external_encoder_count'."""
        return self._external_encoder_count

    @external_encoder_count.setter
    def external_encoder_count(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint16, \
                "The 'external_encoder_count' numpy.ndarray() must have the dtype of 'numpy.uint16'"
            assert value.size == 2, \
                "The 'external_encoder_count' numpy.ndarray() must have a size of 2"
            self._external_encoder_count = value
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
                 len(value) == 2 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 65536 for val in value)), \
                "The 'external_encoder_count' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 65535]"
        self._external_encoder_count = numpy.array(value, dtype=numpy.uint16)

    @builtins.property
    def goal_joint_position(self):
        """Message field 'goal_joint_position'."""
        return self._goal_joint_position

    @goal_joint_position.setter
    def goal_joint_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'goal_joint_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'goal_joint_position' numpy.ndarray() must have a size of 6"
            self._goal_joint_position = value
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
                "The 'goal_joint_position' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._goal_joint_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def goal_tcp_position(self):
        """Message field 'goal_tcp_position'."""
        return self._goal_tcp_position

    @goal_tcp_position.setter
    def goal_tcp_position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'goal_tcp_position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'goal_tcp_position' numpy.ndarray() must have a size of 6"
            self._goal_tcp_position = value
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
                "The 'goal_tcp_position' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._goal_tcp_position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def robot_mode(self):
        """Message field 'robot_mode'."""
        return self._robot_mode

    @robot_mode.setter
    def robot_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'robot_mode' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'robot_mode' field must be an unsigned integer in [0, 255]"
        self._robot_mode = value

    @builtins.property
    def robot_state(self):
        """Message field 'robot_state'."""
        return self._robot_state

    @robot_state.setter
    def robot_state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'robot_state' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'robot_state' field must be an unsigned integer in [0, 255]"
        self._robot_state = value

    @builtins.property
    def control_mode(self):
        """Message field 'control_mode'."""
        return self._control_mode

    @control_mode.setter
    def control_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'control_mode' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'control_mode' field must be an unsigned integer in [0, 65535]"
        self._control_mode = value

    @builtins.property
    def reserved(self):
        """Message field 'reserved'."""
        return self._reserved

    @reserved.setter
    def reserved(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'reserved' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 256, \
                "The 'reserved' numpy.ndarray() must have a size of 256"
            self._reserved = value
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
                 len(value) == 256 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'reserved' field must be a set or sequence with length 256 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._reserved = numpy.array(value, dtype=numpy.uint8)
