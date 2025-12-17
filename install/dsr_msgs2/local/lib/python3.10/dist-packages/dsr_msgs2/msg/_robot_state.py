# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dsr_msgs2:msg/RobotState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'current_posj'
# Member 'current_velj'
# Member 'joint_abs'
# Member 'joint_err'
# Member 'target_posj'
# Member 'target_velj'
# Member 'current_posx'
# Member 'current_tool_posx'
# Member 'current_velx'
# Member 'task_err'
# Member 'target_velx'
# Member 'target_posx'
# Member 'dynamic_tor'
# Member 'actual_jts'
# Member 'actual_ejt'
# Member 'actual_ett'
# Member 'actual_bk'
# Member 'actual_mc'
# Member 'actual_mt'
# Member 'actual_bt'
# Member 'ctrlbox_digital_input'
# Member 'ctrlbox_digital_output'
# Member 'flange_digital_input'
# Member 'flange_digital_output'
# Member 'f_actual_w2b'
# Member 'f_current_vel_world'
# Member 'f_world_ext_target_torque'
# Member 'f_target_pos_world'
# Member 'f_target_vel_world'
# Member 'f_current_vel_user'
# Member 'f_user_ext_task_torque'
# Member 'f_target_pos_user'
# Member 'f_target_vel_user'
# Member 'f_actual_analog_input'
# Member 'i_actual_analog_input_type'
# Member 'f_target_analog_output'
# Member 'i_target_analog_output_type'
# Member 'i_actual_encorder_raw_data'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RobotState(type):
    """Metaclass of message 'RobotState'."""

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
                'dsr_msgs2.msg.RobotState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__robot_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__robot_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__robot_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__robot_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__robot_state

            from dsr_msgs2.msg import ModbusState
            if ModbusState.__class__._TYPE_SUPPORT is None:
                ModbusState.__class__.__import_type_support__()

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


class RobotState(metaclass=Metaclass_RobotState):
    """Message class 'RobotState'."""

    __slots__ = [
        '_robot_state',
        '_robot_state_str',
        '_actual_mode',
        '_actual_space',
        '_current_posj',
        '_current_velj',
        '_joint_abs',
        '_joint_err',
        '_target_posj',
        '_target_velj',
        '_current_posx',
        '_current_tool_posx',
        '_current_velx',
        '_task_err',
        '_target_velx',
        '_target_posx',
        '_dynamic_tor',
        '_actual_jts',
        '_actual_ejt',
        '_actual_ett',
        '_actual_bk',
        '_actual_mc',
        '_actual_mt',
        '_solution_space',
        '_sync_time',
        '_actual_bt',
        '_rotation_matrix',
        '_ctrlbox_digital_input',
        '_ctrlbox_digital_output',
        '_flange_digital_input',
        '_flange_digital_output',
        '_modbus_state',
        '_access_control',
        '_homming_completed',
        '_tp_initialized',
        '_mastering_need',
        '_drl_stopped',
        '_disconnected',
        '_f_actual_w2b',
        '_f_current_pos_world',
        '_f_current_vel_world',
        '_f_world_ext_target_torque',
        '_f_target_pos_world',
        '_f_target_vel_world',
        '_f_rotation_matrix_world',
        '_i_actual_user_coord_num',
        '_i_coord_ref',
        '_f_current_pos_user',
        '_f_current_vel_user',
        '_f_user_ext_task_torque',
        '_f_target_pos_user',
        '_f_target_vel_user',
        '_f_rotation_matrix_user',
        '_f_actual_analog_input',
        '_b_actual_switch_input',
        '_b_actual_safety_input',
        '_i_actual_analog_input_type',
        '_f_target_analog_output',
        '_i_target_analog_output_type',
        '_b_actual_encorder_strove_signal',
        '_i_actual_encorder_raw_data',
        '_b_actual_encorder_reset_signal',
    ]

    _fields_and_field_types = {
        'robot_state': 'int32',
        'robot_state_str': 'string',
        'actual_mode': 'int8',
        'actual_space': 'int8',
        'current_posj': 'double[6]',
        'current_velj': 'double[6]',
        'joint_abs': 'double[6]',
        'joint_err': 'double[6]',
        'target_posj': 'double[6]',
        'target_velj': 'double[6]',
        'current_posx': 'double[6]',
        'current_tool_posx': 'double[6]',
        'current_velx': 'double[6]',
        'task_err': 'double[6]',
        'target_velx': 'double[6]',
        'target_posx': 'double[6]',
        'dynamic_tor': 'double[6]',
        'actual_jts': 'double[6]',
        'actual_ejt': 'double[6]',
        'actual_ett': 'double[6]',
        'actual_bk': 'int8[6]',
        'actual_mc': 'double[6]',
        'actual_mt': 'double[6]',
        'solution_space': 'int8',
        'sync_time': 'double',
        'actual_bt': 'int8[5]',
        'rotation_matrix': 'sequence<std_msgs/Float64MultiArray>',
        'ctrlbox_digital_input': 'int8[16]',
        'ctrlbox_digital_output': 'int8[16]',
        'flange_digital_input': 'int8[6]',
        'flange_digital_output': 'int8[6]',
        'modbus_state': 'sequence<dsr_msgs2/ModbusState>',
        'access_control': 'int32',
        'homming_completed': 'boolean',
        'tp_initialized': 'boolean',
        'mastering_need': 'int8',
        'drl_stopped': 'boolean',
        'disconnected': 'boolean',
        'f_actual_w2b': 'double[6]',
        'f_current_pos_world': 'sequence<std_msgs/Float64MultiArray>',
        'f_current_vel_world': 'double[6]',
        'f_world_ext_target_torque': 'double[6]',
        'f_target_pos_world': 'double[6]',
        'f_target_vel_world': 'double[6]',
        'f_rotation_matrix_world': 'sequence<std_msgs/Float64MultiArray>',
        'i_actual_user_coord_num': 'int8',
        'i_coord_ref': 'int8',
        'f_current_pos_user': 'sequence<std_msgs/Float64MultiArray>',
        'f_current_vel_user': 'double[6]',
        'f_user_ext_task_torque': 'double[6]',
        'f_target_pos_user': 'double[6]',
        'f_target_vel_user': 'double[6]',
        'f_rotation_matrix_user': 'sequence<std_msgs/Float64MultiArray>',
        'f_actual_analog_input': 'double[6]',
        'b_actual_switch_input': 'boolean[3]',
        'b_actual_safety_input': 'boolean[2]',
        'i_actual_analog_input_type': 'int8[2]',
        'f_target_analog_output': 'double[2]',
        'i_target_analog_output_type': 'int8[2]',
        'b_actual_encorder_strove_signal': 'boolean[2]',
        'i_actual_encorder_raw_data': 'int8[2]',
        'b_actual_encorder_reset_signal': 'boolean[2]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
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
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 5),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray')),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 16),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 16),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 6),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['dsr_msgs2', 'msg'], 'ModbusState')),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray')),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray')),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray')),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Float64MultiArray')),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('boolean'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('boolean'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('boolean'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('boolean'), 2),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.robot_state = kwargs.get('robot_state', int())
        self.robot_state_str = kwargs.get('robot_state_str', str())
        self.actual_mode = kwargs.get('actual_mode', int())
        self.actual_space = kwargs.get('actual_space', int())
        if 'current_posj' not in kwargs:
            self.current_posj = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.current_posj = kwargs.get('current_posj')
        if 'current_velj' not in kwargs:
            self.current_velj = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.current_velj = kwargs.get('current_velj')
        if 'joint_abs' not in kwargs:
            self.joint_abs = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.joint_abs = kwargs.get('joint_abs')
        if 'joint_err' not in kwargs:
            self.joint_err = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.joint_err = kwargs.get('joint_err')
        if 'target_posj' not in kwargs:
            self.target_posj = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_posj = kwargs.get('target_posj')
        if 'target_velj' not in kwargs:
            self.target_velj = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_velj = kwargs.get('target_velj')
        if 'current_posx' not in kwargs:
            self.current_posx = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.current_posx = kwargs.get('current_posx')
        if 'current_tool_posx' not in kwargs:
            self.current_tool_posx = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.current_tool_posx = kwargs.get('current_tool_posx')
        if 'current_velx' not in kwargs:
            self.current_velx = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.current_velx = kwargs.get('current_velx')
        if 'task_err' not in kwargs:
            self.task_err = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.task_err = kwargs.get('task_err')
        if 'target_velx' not in kwargs:
            self.target_velx = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_velx = kwargs.get('target_velx')
        if 'target_posx' not in kwargs:
            self.target_posx = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.target_posx = kwargs.get('target_posx')
        if 'dynamic_tor' not in kwargs:
            self.dynamic_tor = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.dynamic_tor = kwargs.get('dynamic_tor')
        if 'actual_jts' not in kwargs:
            self.actual_jts = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_jts = kwargs.get('actual_jts')
        if 'actual_ejt' not in kwargs:
            self.actual_ejt = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_ejt = kwargs.get('actual_ejt')
        if 'actual_ett' not in kwargs:
            self.actual_ett = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_ett = kwargs.get('actual_ett')
        if 'actual_bk' not in kwargs:
            self.actual_bk = numpy.zeros(6, dtype=numpy.int8)
        else:
            self.actual_bk = kwargs.get('actual_bk')
        if 'actual_mc' not in kwargs:
            self.actual_mc = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_mc = kwargs.get('actual_mc')
        if 'actual_mt' not in kwargs:
            self.actual_mt = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.actual_mt = kwargs.get('actual_mt')
        self.solution_space = kwargs.get('solution_space', int())
        self.sync_time = kwargs.get('sync_time', float())
        if 'actual_bt' not in kwargs:
            self.actual_bt = numpy.zeros(5, dtype=numpy.int8)
        else:
            self.actual_bt = kwargs.get('actual_bt')
        self.rotation_matrix = kwargs.get('rotation_matrix', [])
        if 'ctrlbox_digital_input' not in kwargs:
            self.ctrlbox_digital_input = numpy.zeros(16, dtype=numpy.int8)
        else:
            self.ctrlbox_digital_input = kwargs.get('ctrlbox_digital_input')
        if 'ctrlbox_digital_output' not in kwargs:
            self.ctrlbox_digital_output = numpy.zeros(16, dtype=numpy.int8)
        else:
            self.ctrlbox_digital_output = kwargs.get('ctrlbox_digital_output')
        if 'flange_digital_input' not in kwargs:
            self.flange_digital_input = numpy.zeros(6, dtype=numpy.int8)
        else:
            self.flange_digital_input = kwargs.get('flange_digital_input')
        if 'flange_digital_output' not in kwargs:
            self.flange_digital_output = numpy.zeros(6, dtype=numpy.int8)
        else:
            self.flange_digital_output = kwargs.get('flange_digital_output')
        self.modbus_state = kwargs.get('modbus_state', [])
        self.access_control = kwargs.get('access_control', int())
        self.homming_completed = kwargs.get('homming_completed', bool())
        self.tp_initialized = kwargs.get('tp_initialized', bool())
        self.mastering_need = kwargs.get('mastering_need', int())
        self.drl_stopped = kwargs.get('drl_stopped', bool())
        self.disconnected = kwargs.get('disconnected', bool())
        if 'f_actual_w2b' not in kwargs:
            self.f_actual_w2b = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_actual_w2b = kwargs.get('f_actual_w2b')
        self.f_current_pos_world = kwargs.get('f_current_pos_world', [])
        if 'f_current_vel_world' not in kwargs:
            self.f_current_vel_world = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_current_vel_world = kwargs.get('f_current_vel_world')
        if 'f_world_ext_target_torque' not in kwargs:
            self.f_world_ext_target_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_world_ext_target_torque = kwargs.get('f_world_ext_target_torque')
        if 'f_target_pos_world' not in kwargs:
            self.f_target_pos_world = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_target_pos_world = kwargs.get('f_target_pos_world')
        if 'f_target_vel_world' not in kwargs:
            self.f_target_vel_world = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_target_vel_world = kwargs.get('f_target_vel_world')
        self.f_rotation_matrix_world = kwargs.get('f_rotation_matrix_world', [])
        self.i_actual_user_coord_num = kwargs.get('i_actual_user_coord_num', int())
        self.i_coord_ref = kwargs.get('i_coord_ref', int())
        self.f_current_pos_user = kwargs.get('f_current_pos_user', [])
        if 'f_current_vel_user' not in kwargs:
            self.f_current_vel_user = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_current_vel_user = kwargs.get('f_current_vel_user')
        if 'f_user_ext_task_torque' not in kwargs:
            self.f_user_ext_task_torque = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_user_ext_task_torque = kwargs.get('f_user_ext_task_torque')
        if 'f_target_pos_user' not in kwargs:
            self.f_target_pos_user = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_target_pos_user = kwargs.get('f_target_pos_user')
        if 'f_target_vel_user' not in kwargs:
            self.f_target_vel_user = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_target_vel_user = kwargs.get('f_target_vel_user')
        self.f_rotation_matrix_user = kwargs.get('f_rotation_matrix_user', [])
        if 'f_actual_analog_input' not in kwargs:
            self.f_actual_analog_input = numpy.zeros(6, dtype=numpy.float64)
        else:
            self.f_actual_analog_input = kwargs.get('f_actual_analog_input')
        self.b_actual_switch_input = kwargs.get(
            'b_actual_switch_input',
            [bool() for x in range(3)]
        )
        self.b_actual_safety_input = kwargs.get(
            'b_actual_safety_input',
            [bool() for x in range(2)]
        )
        if 'i_actual_analog_input_type' not in kwargs:
            self.i_actual_analog_input_type = numpy.zeros(2, dtype=numpy.int8)
        else:
            self.i_actual_analog_input_type = kwargs.get('i_actual_analog_input_type')
        if 'f_target_analog_output' not in kwargs:
            self.f_target_analog_output = numpy.zeros(2, dtype=numpy.float64)
        else:
            self.f_target_analog_output = kwargs.get('f_target_analog_output')
        if 'i_target_analog_output_type' not in kwargs:
            self.i_target_analog_output_type = numpy.zeros(2, dtype=numpy.int8)
        else:
            self.i_target_analog_output_type = kwargs.get('i_target_analog_output_type')
        self.b_actual_encorder_strove_signal = kwargs.get(
            'b_actual_encorder_strove_signal',
            [bool() for x in range(2)]
        )
        if 'i_actual_encorder_raw_data' not in kwargs:
            self.i_actual_encorder_raw_data = numpy.zeros(2, dtype=numpy.int8)
        else:
            self.i_actual_encorder_raw_data = kwargs.get('i_actual_encorder_raw_data')
        self.b_actual_encorder_reset_signal = kwargs.get(
            'b_actual_encorder_reset_signal',
            [bool() for x in range(2)]
        )

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
        if self.robot_state != other.robot_state:
            return False
        if self.robot_state_str != other.robot_state_str:
            return False
        if self.actual_mode != other.actual_mode:
            return False
        if self.actual_space != other.actual_space:
            return False
        if any(self.current_posj != other.current_posj):
            return False
        if any(self.current_velj != other.current_velj):
            return False
        if any(self.joint_abs != other.joint_abs):
            return False
        if any(self.joint_err != other.joint_err):
            return False
        if any(self.target_posj != other.target_posj):
            return False
        if any(self.target_velj != other.target_velj):
            return False
        if any(self.current_posx != other.current_posx):
            return False
        if any(self.current_tool_posx != other.current_tool_posx):
            return False
        if any(self.current_velx != other.current_velx):
            return False
        if any(self.task_err != other.task_err):
            return False
        if any(self.target_velx != other.target_velx):
            return False
        if any(self.target_posx != other.target_posx):
            return False
        if any(self.dynamic_tor != other.dynamic_tor):
            return False
        if any(self.actual_jts != other.actual_jts):
            return False
        if any(self.actual_ejt != other.actual_ejt):
            return False
        if any(self.actual_ett != other.actual_ett):
            return False
        if any(self.actual_bk != other.actual_bk):
            return False
        if any(self.actual_mc != other.actual_mc):
            return False
        if any(self.actual_mt != other.actual_mt):
            return False
        if self.solution_space != other.solution_space:
            return False
        if self.sync_time != other.sync_time:
            return False
        if any(self.actual_bt != other.actual_bt):
            return False
        if self.rotation_matrix != other.rotation_matrix:
            return False
        if any(self.ctrlbox_digital_input != other.ctrlbox_digital_input):
            return False
        if any(self.ctrlbox_digital_output != other.ctrlbox_digital_output):
            return False
        if any(self.flange_digital_input != other.flange_digital_input):
            return False
        if any(self.flange_digital_output != other.flange_digital_output):
            return False
        if self.modbus_state != other.modbus_state:
            return False
        if self.access_control != other.access_control:
            return False
        if self.homming_completed != other.homming_completed:
            return False
        if self.tp_initialized != other.tp_initialized:
            return False
        if self.mastering_need != other.mastering_need:
            return False
        if self.drl_stopped != other.drl_stopped:
            return False
        if self.disconnected != other.disconnected:
            return False
        if any(self.f_actual_w2b != other.f_actual_w2b):
            return False
        if self.f_current_pos_world != other.f_current_pos_world:
            return False
        if any(self.f_current_vel_world != other.f_current_vel_world):
            return False
        if any(self.f_world_ext_target_torque != other.f_world_ext_target_torque):
            return False
        if any(self.f_target_pos_world != other.f_target_pos_world):
            return False
        if any(self.f_target_vel_world != other.f_target_vel_world):
            return False
        if self.f_rotation_matrix_world != other.f_rotation_matrix_world:
            return False
        if self.i_actual_user_coord_num != other.i_actual_user_coord_num:
            return False
        if self.i_coord_ref != other.i_coord_ref:
            return False
        if self.f_current_pos_user != other.f_current_pos_user:
            return False
        if any(self.f_current_vel_user != other.f_current_vel_user):
            return False
        if any(self.f_user_ext_task_torque != other.f_user_ext_task_torque):
            return False
        if any(self.f_target_pos_user != other.f_target_pos_user):
            return False
        if any(self.f_target_vel_user != other.f_target_vel_user):
            return False
        if self.f_rotation_matrix_user != other.f_rotation_matrix_user:
            return False
        if any(self.f_actual_analog_input != other.f_actual_analog_input):
            return False
        if self.b_actual_switch_input != other.b_actual_switch_input:
            return False
        if self.b_actual_safety_input != other.b_actual_safety_input:
            return False
        if any(self.i_actual_analog_input_type != other.i_actual_analog_input_type):
            return False
        if any(self.f_target_analog_output != other.f_target_analog_output):
            return False
        if any(self.i_target_analog_output_type != other.i_target_analog_output_type):
            return False
        if self.b_actual_encorder_strove_signal != other.b_actual_encorder_strove_signal:
            return False
        if any(self.i_actual_encorder_raw_data != other.i_actual_encorder_raw_data):
            return False
        if self.b_actual_encorder_reset_signal != other.b_actual_encorder_reset_signal:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

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
            assert value >= -2147483648 and value < 2147483648, \
                "The 'robot_state' field must be an integer in [-2147483648, 2147483647]"
        self._robot_state = value

    @builtins.property
    def robot_state_str(self):
        """Message field 'robot_state_str'."""
        return self._robot_state_str

    @robot_state_str.setter
    def robot_state_str(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'robot_state_str' field must be of type 'str'"
        self._robot_state_str = value

    @builtins.property
    def actual_mode(self):
        """Message field 'actual_mode'."""
        return self._actual_mode

    @actual_mode.setter
    def actual_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'actual_mode' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'actual_mode' field must be an integer in [-128, 127]"
        self._actual_mode = value

    @builtins.property
    def actual_space(self):
        """Message field 'actual_space'."""
        return self._actual_space

    @actual_space.setter
    def actual_space(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'actual_space' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'actual_space' field must be an integer in [-128, 127]"
        self._actual_space = value

    @builtins.property
    def current_posj(self):
        """Message field 'current_posj'."""
        return self._current_posj

    @current_posj.setter
    def current_posj(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'current_posj' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'current_posj' numpy.ndarray() must have a size of 6"
            self._current_posj = value
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
                "The 'current_posj' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._current_posj = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def current_velj(self):
        """Message field 'current_velj'."""
        return self._current_velj

    @current_velj.setter
    def current_velj(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'current_velj' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'current_velj' numpy.ndarray() must have a size of 6"
            self._current_velj = value
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
                "The 'current_velj' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._current_velj = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joint_abs(self):
        """Message field 'joint_abs'."""
        return self._joint_abs

    @joint_abs.setter
    def joint_abs(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'joint_abs' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'joint_abs' numpy.ndarray() must have a size of 6"
            self._joint_abs = value
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
                "The 'joint_abs' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joint_abs = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def joint_err(self):
        """Message field 'joint_err'."""
        return self._joint_err

    @joint_err.setter
    def joint_err(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'joint_err' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'joint_err' numpy.ndarray() must have a size of 6"
            self._joint_err = value
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
                "The 'joint_err' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._joint_err = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_posj(self):
        """Message field 'target_posj'."""
        return self._target_posj

    @target_posj.setter
    def target_posj(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_posj' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_posj' numpy.ndarray() must have a size of 6"
            self._target_posj = value
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
                "The 'target_posj' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_posj = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_velj(self):
        """Message field 'target_velj'."""
        return self._target_velj

    @target_velj.setter
    def target_velj(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_velj' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_velj' numpy.ndarray() must have a size of 6"
            self._target_velj = value
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
                "The 'target_velj' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_velj = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def current_posx(self):
        """Message field 'current_posx'."""
        return self._current_posx

    @current_posx.setter
    def current_posx(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'current_posx' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'current_posx' numpy.ndarray() must have a size of 6"
            self._current_posx = value
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
                "The 'current_posx' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._current_posx = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def current_tool_posx(self):
        """Message field 'current_tool_posx'."""
        return self._current_tool_posx

    @current_tool_posx.setter
    def current_tool_posx(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'current_tool_posx' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'current_tool_posx' numpy.ndarray() must have a size of 6"
            self._current_tool_posx = value
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
                "The 'current_tool_posx' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._current_tool_posx = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def current_velx(self):
        """Message field 'current_velx'."""
        return self._current_velx

    @current_velx.setter
    def current_velx(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'current_velx' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'current_velx' numpy.ndarray() must have a size of 6"
            self._current_velx = value
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
                "The 'current_velx' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._current_velx = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def task_err(self):
        """Message field 'task_err'."""
        return self._task_err

    @task_err.setter
    def task_err(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'task_err' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'task_err' numpy.ndarray() must have a size of 6"
            self._task_err = value
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
                "The 'task_err' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._task_err = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_velx(self):
        """Message field 'target_velx'."""
        return self._target_velx

    @target_velx.setter
    def target_velx(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_velx' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_velx' numpy.ndarray() must have a size of 6"
            self._target_velx = value
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
                "The 'target_velx' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_velx = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def target_posx(self):
        """Message field 'target_posx'."""
        return self._target_posx

    @target_posx.setter
    def target_posx(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'target_posx' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'target_posx' numpy.ndarray() must have a size of 6"
            self._target_posx = value
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
                "The 'target_posx' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._target_posx = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def dynamic_tor(self):
        """Message field 'dynamic_tor'."""
        return self._dynamic_tor

    @dynamic_tor.setter
    def dynamic_tor(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'dynamic_tor' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'dynamic_tor' numpy.ndarray() must have a size of 6"
            self._dynamic_tor = value
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
                "The 'dynamic_tor' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._dynamic_tor = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_jts(self):
        """Message field 'actual_jts'."""
        return self._actual_jts

    @actual_jts.setter
    def actual_jts(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_jts' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_jts' numpy.ndarray() must have a size of 6"
            self._actual_jts = value
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
                "The 'actual_jts' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_jts = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_ejt(self):
        """Message field 'actual_ejt'."""
        return self._actual_ejt

    @actual_ejt.setter
    def actual_ejt(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_ejt' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_ejt' numpy.ndarray() must have a size of 6"
            self._actual_ejt = value
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
                "The 'actual_ejt' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_ejt = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_ett(self):
        """Message field 'actual_ett'."""
        return self._actual_ett

    @actual_ett.setter
    def actual_ett(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_ett' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_ett' numpy.ndarray() must have a size of 6"
            self._actual_ett = value
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
                "The 'actual_ett' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_ett = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_bk(self):
        """Message field 'actual_bk'."""
        return self._actual_bk

    @actual_bk.setter
    def actual_bk(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'actual_bk' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 6, \
                "The 'actual_bk' numpy.ndarray() must have a size of 6"
            self._actual_bk = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'actual_bk' field must be a set or sequence with length 6 and each value of type 'int' and each integer in [-128, 127]"
        self._actual_bk = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def actual_mc(self):
        """Message field 'actual_mc'."""
        return self._actual_mc

    @actual_mc.setter
    def actual_mc(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_mc' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_mc' numpy.ndarray() must have a size of 6"
            self._actual_mc = value
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
                "The 'actual_mc' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_mc = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def actual_mt(self):
        """Message field 'actual_mt'."""
        return self._actual_mt

    @actual_mt.setter
    def actual_mt(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'actual_mt' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'actual_mt' numpy.ndarray() must have a size of 6"
            self._actual_mt = value
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
                "The 'actual_mt' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._actual_mt = numpy.array(value, dtype=numpy.float64)

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
            assert value >= -128 and value < 128, \
                "The 'solution_space' field must be an integer in [-128, 127]"
        self._solution_space = value

    @builtins.property
    def sync_time(self):
        """Message field 'sync_time'."""
        return self._sync_time

    @sync_time.setter
    def sync_time(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'sync_time' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'sync_time' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._sync_time = value

    @builtins.property
    def actual_bt(self):
        """Message field 'actual_bt'."""
        return self._actual_bt

    @actual_bt.setter
    def actual_bt(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'actual_bt' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 5, \
                "The 'actual_bt' numpy.ndarray() must have a size of 5"
            self._actual_bt = value
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
                 len(value) == 5 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'actual_bt' field must be a set or sequence with length 5 and each value of type 'int' and each integer in [-128, 127]"
        self._actual_bt = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def rotation_matrix(self):
        """Message field 'rotation_matrix'."""
        return self._rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, value):
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
                "The 'rotation_matrix' field must be a set or sequence and each value of type 'Float64MultiArray'"
        self._rotation_matrix = value

    @builtins.property
    def ctrlbox_digital_input(self):
        """Message field 'ctrlbox_digital_input'."""
        return self._ctrlbox_digital_input

    @ctrlbox_digital_input.setter
    def ctrlbox_digital_input(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'ctrlbox_digital_input' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 16, \
                "The 'ctrlbox_digital_input' numpy.ndarray() must have a size of 16"
            self._ctrlbox_digital_input = value
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
                 len(value) == 16 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'ctrlbox_digital_input' field must be a set or sequence with length 16 and each value of type 'int' and each integer in [-128, 127]"
        self._ctrlbox_digital_input = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def ctrlbox_digital_output(self):
        """Message field 'ctrlbox_digital_output'."""
        return self._ctrlbox_digital_output

    @ctrlbox_digital_output.setter
    def ctrlbox_digital_output(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'ctrlbox_digital_output' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 16, \
                "The 'ctrlbox_digital_output' numpy.ndarray() must have a size of 16"
            self._ctrlbox_digital_output = value
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
                 len(value) == 16 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'ctrlbox_digital_output' field must be a set or sequence with length 16 and each value of type 'int' and each integer in [-128, 127]"
        self._ctrlbox_digital_output = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def flange_digital_input(self):
        """Message field 'flange_digital_input'."""
        return self._flange_digital_input

    @flange_digital_input.setter
    def flange_digital_input(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'flange_digital_input' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 6, \
                "The 'flange_digital_input' numpy.ndarray() must have a size of 6"
            self._flange_digital_input = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'flange_digital_input' field must be a set or sequence with length 6 and each value of type 'int' and each integer in [-128, 127]"
        self._flange_digital_input = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def flange_digital_output(self):
        """Message field 'flange_digital_output'."""
        return self._flange_digital_output

    @flange_digital_output.setter
    def flange_digital_output(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'flange_digital_output' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 6, \
                "The 'flange_digital_output' numpy.ndarray() must have a size of 6"
            self._flange_digital_output = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'flange_digital_output' field must be a set or sequence with length 6 and each value of type 'int' and each integer in [-128, 127]"
        self._flange_digital_output = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def modbus_state(self):
        """Message field 'modbus_state'."""
        return self._modbus_state

    @modbus_state.setter
    def modbus_state(self, value):
        if __debug__:
            from dsr_msgs2.msg import ModbusState
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
                 all(isinstance(v, ModbusState) for v in value) and
                 True), \
                "The 'modbus_state' field must be a set or sequence and each value of type 'ModbusState'"
        self._modbus_state = value

    @builtins.property
    def access_control(self):
        """Message field 'access_control'."""
        return self._access_control

    @access_control.setter
    def access_control(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'access_control' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'access_control' field must be an integer in [-2147483648, 2147483647]"
        self._access_control = value

    @builtins.property
    def homming_completed(self):
        """Message field 'homming_completed'."""
        return self._homming_completed

    @homming_completed.setter
    def homming_completed(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'homming_completed' field must be of type 'bool'"
        self._homming_completed = value

    @builtins.property
    def tp_initialized(self):
        """Message field 'tp_initialized'."""
        return self._tp_initialized

    @tp_initialized.setter
    def tp_initialized(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'tp_initialized' field must be of type 'bool'"
        self._tp_initialized = value

    @builtins.property
    def mastering_need(self):
        """Message field 'mastering_need'."""
        return self._mastering_need

    @mastering_need.setter
    def mastering_need(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mastering_need' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'mastering_need' field must be an integer in [-128, 127]"
        self._mastering_need = value

    @builtins.property
    def drl_stopped(self):
        """Message field 'drl_stopped'."""
        return self._drl_stopped

    @drl_stopped.setter
    def drl_stopped(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'drl_stopped' field must be of type 'bool'"
        self._drl_stopped = value

    @builtins.property
    def disconnected(self):
        """Message field 'disconnected'."""
        return self._disconnected

    @disconnected.setter
    def disconnected(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'disconnected' field must be of type 'bool'"
        self._disconnected = value

    @builtins.property
    def f_actual_w2b(self):
        """Message field 'f_actual_w2b'."""
        return self._f_actual_w2b

    @f_actual_w2b.setter
    def f_actual_w2b(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_actual_w2b' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_actual_w2b' numpy.ndarray() must have a size of 6"
            self._f_actual_w2b = value
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
                "The 'f_actual_w2b' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_actual_w2b = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_current_pos_world(self):
        """Message field 'f_current_pos_world'."""
        return self._f_current_pos_world

    @f_current_pos_world.setter
    def f_current_pos_world(self, value):
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
                "The 'f_current_pos_world' field must be a set or sequence and each value of type 'Float64MultiArray'"
        self._f_current_pos_world = value

    @builtins.property
    def f_current_vel_world(self):
        """Message field 'f_current_vel_world'."""
        return self._f_current_vel_world

    @f_current_vel_world.setter
    def f_current_vel_world(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_current_vel_world' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_current_vel_world' numpy.ndarray() must have a size of 6"
            self._f_current_vel_world = value
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
                "The 'f_current_vel_world' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_current_vel_world = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_world_ext_target_torque(self):
        """Message field 'f_world_ext_target_torque'."""
        return self._f_world_ext_target_torque

    @f_world_ext_target_torque.setter
    def f_world_ext_target_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_world_ext_target_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_world_ext_target_torque' numpy.ndarray() must have a size of 6"
            self._f_world_ext_target_torque = value
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
                "The 'f_world_ext_target_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_world_ext_target_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_target_pos_world(self):
        """Message field 'f_target_pos_world'."""
        return self._f_target_pos_world

    @f_target_pos_world.setter
    def f_target_pos_world(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_target_pos_world' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_target_pos_world' numpy.ndarray() must have a size of 6"
            self._f_target_pos_world = value
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
                "The 'f_target_pos_world' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_target_pos_world = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_target_vel_world(self):
        """Message field 'f_target_vel_world'."""
        return self._f_target_vel_world

    @f_target_vel_world.setter
    def f_target_vel_world(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_target_vel_world' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_target_vel_world' numpy.ndarray() must have a size of 6"
            self._f_target_vel_world = value
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
                "The 'f_target_vel_world' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_target_vel_world = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_rotation_matrix_world(self):
        """Message field 'f_rotation_matrix_world'."""
        return self._f_rotation_matrix_world

    @f_rotation_matrix_world.setter
    def f_rotation_matrix_world(self, value):
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
                "The 'f_rotation_matrix_world' field must be a set or sequence and each value of type 'Float64MultiArray'"
        self._f_rotation_matrix_world = value

    @builtins.property
    def i_actual_user_coord_num(self):
        """Message field 'i_actual_user_coord_num'."""
        return self._i_actual_user_coord_num

    @i_actual_user_coord_num.setter
    def i_actual_user_coord_num(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'i_actual_user_coord_num' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'i_actual_user_coord_num' field must be an integer in [-128, 127]"
        self._i_actual_user_coord_num = value

    @builtins.property
    def i_coord_ref(self):
        """Message field 'i_coord_ref'."""
        return self._i_coord_ref

    @i_coord_ref.setter
    def i_coord_ref(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'i_coord_ref' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'i_coord_ref' field must be an integer in [-128, 127]"
        self._i_coord_ref = value

    @builtins.property
    def f_current_pos_user(self):
        """Message field 'f_current_pos_user'."""
        return self._f_current_pos_user

    @f_current_pos_user.setter
    def f_current_pos_user(self, value):
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
                "The 'f_current_pos_user' field must be a set or sequence and each value of type 'Float64MultiArray'"
        self._f_current_pos_user = value

    @builtins.property
    def f_current_vel_user(self):
        """Message field 'f_current_vel_user'."""
        return self._f_current_vel_user

    @f_current_vel_user.setter
    def f_current_vel_user(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_current_vel_user' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_current_vel_user' numpy.ndarray() must have a size of 6"
            self._f_current_vel_user = value
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
                "The 'f_current_vel_user' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_current_vel_user = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_user_ext_task_torque(self):
        """Message field 'f_user_ext_task_torque'."""
        return self._f_user_ext_task_torque

    @f_user_ext_task_torque.setter
    def f_user_ext_task_torque(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_user_ext_task_torque' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_user_ext_task_torque' numpy.ndarray() must have a size of 6"
            self._f_user_ext_task_torque = value
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
                "The 'f_user_ext_task_torque' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_user_ext_task_torque = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_target_pos_user(self):
        """Message field 'f_target_pos_user'."""
        return self._f_target_pos_user

    @f_target_pos_user.setter
    def f_target_pos_user(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_target_pos_user' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_target_pos_user' numpy.ndarray() must have a size of 6"
            self._f_target_pos_user = value
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
                "The 'f_target_pos_user' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_target_pos_user = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_target_vel_user(self):
        """Message field 'f_target_vel_user'."""
        return self._f_target_vel_user

    @f_target_vel_user.setter
    def f_target_vel_user(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_target_vel_user' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_target_vel_user' numpy.ndarray() must have a size of 6"
            self._f_target_vel_user = value
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
                "The 'f_target_vel_user' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_target_vel_user = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def f_rotation_matrix_user(self):
        """Message field 'f_rotation_matrix_user'."""
        return self._f_rotation_matrix_user

    @f_rotation_matrix_user.setter
    def f_rotation_matrix_user(self, value):
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
                "The 'f_rotation_matrix_user' field must be a set or sequence and each value of type 'Float64MultiArray'"
        self._f_rotation_matrix_user = value

    @builtins.property
    def f_actual_analog_input(self):
        """Message field 'f_actual_analog_input'."""
        return self._f_actual_analog_input

    @f_actual_analog_input.setter
    def f_actual_analog_input(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_actual_analog_input' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 6, \
                "The 'f_actual_analog_input' numpy.ndarray() must have a size of 6"
            self._f_actual_analog_input = value
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
                "The 'f_actual_analog_input' field must be a set or sequence with length 6 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_actual_analog_input = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def b_actual_switch_input(self):
        """Message field 'b_actual_switch_input'."""
        return self._b_actual_switch_input

    @b_actual_switch_input.setter
    def b_actual_switch_input(self, value):
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
                 len(value) == 3 and
                 all(isinstance(v, bool) for v in value) and
                 True), \
                "The 'b_actual_switch_input' field must be a set or sequence with length 3 and each value of type 'bool'"
        self._b_actual_switch_input = value

    @builtins.property
    def b_actual_safety_input(self):
        """Message field 'b_actual_safety_input'."""
        return self._b_actual_safety_input

    @b_actual_safety_input.setter
    def b_actual_safety_input(self, value):
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
                 all(isinstance(v, bool) for v in value) and
                 True), \
                "The 'b_actual_safety_input' field must be a set or sequence with length 2 and each value of type 'bool'"
        self._b_actual_safety_input = value

    @builtins.property
    def i_actual_analog_input_type(self):
        """Message field 'i_actual_analog_input_type'."""
        return self._i_actual_analog_input_type

    @i_actual_analog_input_type.setter
    def i_actual_analog_input_type(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'i_actual_analog_input_type' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 2, \
                "The 'i_actual_analog_input_type' numpy.ndarray() must have a size of 2"
            self._i_actual_analog_input_type = value
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
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'i_actual_analog_input_type' field must be a set or sequence with length 2 and each value of type 'int' and each integer in [-128, 127]"
        self._i_actual_analog_input_type = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def f_target_analog_output(self):
        """Message field 'f_target_analog_output'."""
        return self._f_target_analog_output

    @f_target_analog_output.setter
    def f_target_analog_output(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'f_target_analog_output' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 2, \
                "The 'f_target_analog_output' numpy.ndarray() must have a size of 2"
            self._f_target_analog_output = value
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
                "The 'f_target_analog_output' field must be a set or sequence with length 2 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._f_target_analog_output = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def i_target_analog_output_type(self):
        """Message field 'i_target_analog_output_type'."""
        return self._i_target_analog_output_type

    @i_target_analog_output_type.setter
    def i_target_analog_output_type(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'i_target_analog_output_type' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 2, \
                "The 'i_target_analog_output_type' numpy.ndarray() must have a size of 2"
            self._i_target_analog_output_type = value
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
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'i_target_analog_output_type' field must be a set or sequence with length 2 and each value of type 'int' and each integer in [-128, 127]"
        self._i_target_analog_output_type = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def b_actual_encorder_strove_signal(self):
        """Message field 'b_actual_encorder_strove_signal'."""
        return self._b_actual_encorder_strove_signal

    @b_actual_encorder_strove_signal.setter
    def b_actual_encorder_strove_signal(self, value):
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
                 all(isinstance(v, bool) for v in value) and
                 True), \
                "The 'b_actual_encorder_strove_signal' field must be a set or sequence with length 2 and each value of type 'bool'"
        self._b_actual_encorder_strove_signal = value

    @builtins.property
    def i_actual_encorder_raw_data(self):
        """Message field 'i_actual_encorder_raw_data'."""
        return self._i_actual_encorder_raw_data

    @i_actual_encorder_raw_data.setter
    def i_actual_encorder_raw_data(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'i_actual_encorder_raw_data' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 2, \
                "The 'i_actual_encorder_raw_data' numpy.ndarray() must have a size of 2"
            self._i_actual_encorder_raw_data = value
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
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'i_actual_encorder_raw_data' field must be a set or sequence with length 2 and each value of type 'int' and each integer in [-128, 127]"
        self._i_actual_encorder_raw_data = numpy.array(value, dtype=numpy.int8)

    @builtins.property
    def b_actual_encorder_reset_signal(self):
        """Message field 'b_actual_encorder_reset_signal'."""
        return self._b_actual_encorder_reset_signal

    @b_actual_encorder_reset_signal.setter
    def b_actual_encorder_reset_signal(self, value):
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
                 all(isinstance(v, bool) for v in value) and
                 True), \
                "The 'b_actual_encorder_reset_signal' field must be a set or sequence with length 2 and each value of type 'bool'"
        self._b_actual_encorder_reset_signal = value
