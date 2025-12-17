// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/MoveSplineJoint.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_SPLINE_JOINT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_SPLINE_JOINT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/move_spline_joint__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MoveSplineJoint_Request_sync_type
{
public:
  explicit Init_MoveSplineJoint_Request_sync_type(::dsr_msgs2::srv::MoveSplineJoint_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::MoveSplineJoint_Request sync_type(::dsr_msgs2::srv::MoveSplineJoint_Request::_sync_type_type arg)
  {
    msg_.sync_type = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSplineJoint_Request msg_;
};

class Init_MoveSplineJoint_Request_mode
{
public:
  explicit Init_MoveSplineJoint_Request_mode(::dsr_msgs2::srv::MoveSplineJoint_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSplineJoint_Request_sync_type mode(::dsr_msgs2::srv::MoveSplineJoint_Request::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_MoveSplineJoint_Request_sync_type(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSplineJoint_Request msg_;
};

class Init_MoveSplineJoint_Request_time
{
public:
  explicit Init_MoveSplineJoint_Request_time(::dsr_msgs2::srv::MoveSplineJoint_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSplineJoint_Request_mode time(::dsr_msgs2::srv::MoveSplineJoint_Request::_time_type arg)
  {
    msg_.time = std::move(arg);
    return Init_MoveSplineJoint_Request_mode(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSplineJoint_Request msg_;
};

class Init_MoveSplineJoint_Request_acc
{
public:
  explicit Init_MoveSplineJoint_Request_acc(::dsr_msgs2::srv::MoveSplineJoint_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSplineJoint_Request_time acc(::dsr_msgs2::srv::MoveSplineJoint_Request::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_MoveSplineJoint_Request_time(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSplineJoint_Request msg_;
};

class Init_MoveSplineJoint_Request_vel
{
public:
  explicit Init_MoveSplineJoint_Request_vel(::dsr_msgs2::srv::MoveSplineJoint_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSplineJoint_Request_acc vel(::dsr_msgs2::srv::MoveSplineJoint_Request::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_MoveSplineJoint_Request_acc(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSplineJoint_Request msg_;
};

class Init_MoveSplineJoint_Request_pos_cnt
{
public:
  explicit Init_MoveSplineJoint_Request_pos_cnt(::dsr_msgs2::srv::MoveSplineJoint_Request & msg)
  : msg_(msg)
  {}
  Init_MoveSplineJoint_Request_vel pos_cnt(::dsr_msgs2::srv::MoveSplineJoint_Request::_pos_cnt_type arg)
  {
    msg_.pos_cnt = std::move(arg);
    return Init_MoveSplineJoint_Request_vel(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSplineJoint_Request msg_;
};

class Init_MoveSplineJoint_Request_pos
{
public:
  Init_MoveSplineJoint_Request_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MoveSplineJoint_Request_pos_cnt pos(::dsr_msgs2::srv::MoveSplineJoint_Request::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_MoveSplineJoint_Request_pos_cnt(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSplineJoint_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MoveSplineJoint_Request>()
{
  return dsr_msgs2::srv::builder::Init_MoveSplineJoint_Request_pos();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MoveSplineJoint_Response_success
{
public:
  Init_MoveSplineJoint_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::MoveSplineJoint_Response success(::dsr_msgs2::srv::MoveSplineJoint_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveSplineJoint_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MoveSplineJoint_Response>()
{
  return dsr_msgs2::srv::builder::Init_MoveSplineJoint_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_SPLINE_JOINT__BUILDER_HPP_
