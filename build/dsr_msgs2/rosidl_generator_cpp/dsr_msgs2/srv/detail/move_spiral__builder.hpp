// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/MoveSpiral.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/move_spiral__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MoveSpiral_Request_sync_type
{
public:
  explicit Init_MoveSpiral_Request_sync_type(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::MoveSpiral_Request sync_type(::dsr_msgs2::srv::MoveSpiral_Request::_sync_type_type arg)
  {
    msg_.sync_type = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_rot_dir
{
public:
  explicit Init_MoveSpiral_Request_rot_dir(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_sync_type rot_dir(::dsr_msgs2::srv::MoveSpiral_Request::_rot_dir_type arg)
  {
    msg_.rot_dir = std::move(arg);
    return Init_MoveSpiral_Request_sync_type(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_spiral_dir
{
public:
  explicit Init_MoveSpiral_Request_spiral_dir(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_rot_dir spiral_dir(::dsr_msgs2::srv::MoveSpiral_Request::_spiral_dir_type arg)
  {
    msg_.spiral_dir = std::move(arg);
    return Init_MoveSpiral_Request_rot_dir(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_mode
{
public:
  explicit Init_MoveSpiral_Request_mode(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_spiral_dir mode(::dsr_msgs2::srv::MoveSpiral_Request::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_MoveSpiral_Request_spiral_dir(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_ref
{
public:
  explicit Init_MoveSpiral_Request_ref(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_mode ref(::dsr_msgs2::srv::MoveSpiral_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_MoveSpiral_Request_mode(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_task_axis
{
public:
  explicit Init_MoveSpiral_Request_task_axis(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_ref task_axis(::dsr_msgs2::srv::MoveSpiral_Request::_task_axis_type arg)
  {
    msg_.task_axis = std::move(arg);
    return Init_MoveSpiral_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_time
{
public:
  explicit Init_MoveSpiral_Request_time(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_task_axis time(::dsr_msgs2::srv::MoveSpiral_Request::_time_type arg)
  {
    msg_.time = std::move(arg);
    return Init_MoveSpiral_Request_task_axis(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_acc
{
public:
  explicit Init_MoveSpiral_Request_acc(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_time acc(::dsr_msgs2::srv::MoveSpiral_Request::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_MoveSpiral_Request_time(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_vel
{
public:
  explicit Init_MoveSpiral_Request_vel(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_acc vel(::dsr_msgs2::srv::MoveSpiral_Request::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_MoveSpiral_Request_acc(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_target_pos
{
public:
  explicit Init_MoveSpiral_Request_target_pos(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_vel target_pos(::dsr_msgs2::srv::MoveSpiral_Request::_target_pos_type arg)
  {
    msg_.target_pos = std::move(arg);
    return Init_MoveSpiral_Request_vel(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_max_length
{
public:
  explicit Init_MoveSpiral_Request_max_length(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_target_pos max_length(::dsr_msgs2::srv::MoveSpiral_Request::_max_length_type arg)
  {
    msg_.max_length = std::move(arg);
    return Init_MoveSpiral_Request_target_pos(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_max_radius
{
public:
  explicit Init_MoveSpiral_Request_max_radius(::dsr_msgs2::srv::MoveSpiral_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSpiral_Request_max_length max_radius(::dsr_msgs2::srv::MoveSpiral_Request::_max_radius_type arg)
  {
    msg_.max_radius = std::move(arg);
    return Init_MoveSpiral_Request_max_length(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

class Init_MoveSpiral_Request_revolution
{
public:
  Init_MoveSpiral_Request_revolution()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MoveSpiral_Request_max_radius revolution(::dsr_msgs2::srv::MoveSpiral_Request::_revolution_type arg)
  {
    msg_.revolution = std::move(arg);
    return Init_MoveSpiral_Request_max_radius(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MoveSpiral_Request>()
{
  return dsr_msgs2::srv::builder::Init_MoveSpiral_Request_revolution();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MoveSpiral_Response_success
{
public:
  Init_MoveSpiral_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::MoveSpiral_Response success(::dsr_msgs2::srv::MoveSpiral_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSpiral_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MoveSpiral_Response>()
{
  return dsr_msgs2::srv::builder::Init_MoveSpiral_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__BUILDER_HPP_
