// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/Jog.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__JOG__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__JOG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/jog__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_Jog_Request_speed
{
public:
  explicit Init_Jog_Request_speed(::dsr_msgs2::srv::Jog_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::Jog_Request speed(::dsr_msgs2::srv::Jog_Request::_speed_type arg)
  {
    msg_.speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::Jog_Request msg_;
};

class Init_Jog_Request_move_reference
{
public:
  explicit Init_Jog_Request_move_reference(::dsr_msgs2::srv::Jog_Request & msg)
  : msg_(msg)
  {}
  Init_Jog_Request_speed move_reference(::dsr_msgs2::srv::Jog_Request::_move_reference_type arg)
  {
    msg_.move_reference = std::move(arg);
    return Init_Jog_Request_speed(msg_);
  }

private:
  ::dsr_msgs2::srv::Jog_Request msg_;
};

class Init_Jog_Request_jog_axis
{
public:
  Init_Jog_Request_jog_axis()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Jog_Request_move_reference jog_axis(::dsr_msgs2::srv::Jog_Request::_jog_axis_type arg)
  {
    msg_.jog_axis = std::move(arg);
    return Init_Jog_Request_move_reference(msg_);
  }

private:
  ::dsr_msgs2::srv::Jog_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::Jog_Request>()
{
  return dsr_msgs2::srv::builder::Init_Jog_Request_jog_axis();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_Jog_Response_success
{
public:
  Init_Jog_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::Jog_Response success(::dsr_msgs2::srv::Jog_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::Jog_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::Jog_Response>()
{
  return dsr_msgs2::srv::builder::Init_Jog_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__JOG__BUILDER_HPP_
