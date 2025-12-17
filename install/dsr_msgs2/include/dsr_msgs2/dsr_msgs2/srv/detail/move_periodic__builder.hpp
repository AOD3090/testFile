// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/MovePeriodic.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_PERIODIC__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_PERIODIC__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/move_periodic__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MovePeriodic_Request_sync_type
{
public:
  explicit Init_MovePeriodic_Request_sync_type(::dsr_msgs2::srv::MovePeriodic_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::MovePeriodic_Request sync_type(::dsr_msgs2::srv::MovePeriodic_Request::_sync_type_type arg)
  {
    msg_.sync_type = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MovePeriodic_Request msg_;
};

class Init_MovePeriodic_Request_ref
{
public:
  explicit Init_MovePeriodic_Request_ref(::dsr_msgs2::srv::MovePeriodic_Request & msg)
  : msg_(msg)
  {}
  Init_MovePeriodic_Request_sync_type ref(::dsr_msgs2::srv::MovePeriodic_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_MovePeriodic_Request_sync_type(msg_);
  }

private:
  ::dsr_msgs2::srv::MovePeriodic_Request msg_;
};

class Init_MovePeriodic_Request_repeat
{
public:
  explicit Init_MovePeriodic_Request_repeat(::dsr_msgs2::srv::MovePeriodic_Request & msg)
  : msg_(msg)
  {}
  Init_MovePeriodic_Request_ref repeat(::dsr_msgs2::srv::MovePeriodic_Request::_repeat_type arg)
  {
    msg_.repeat = std::move(arg);
    return Init_MovePeriodic_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::MovePeriodic_Request msg_;
};

class Init_MovePeriodic_Request_acc
{
public:
  explicit Init_MovePeriodic_Request_acc(::dsr_msgs2::srv::MovePeriodic_Request & msg)
  : msg_(msg)
  {}
  Init_MovePeriodic_Request_repeat acc(::dsr_msgs2::srv::MovePeriodic_Request::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_MovePeriodic_Request_repeat(msg_);
  }

private:
  ::dsr_msgs2::srv::MovePeriodic_Request msg_;
};

class Init_MovePeriodic_Request_periodic
{
public:
  explicit Init_MovePeriodic_Request_periodic(::dsr_msgs2::srv::MovePeriodic_Request & msg)
  : msg_(msg)
  {}
  Init_MovePeriodic_Request_acc periodic(::dsr_msgs2::srv::MovePeriodic_Request::_periodic_type arg)
  {
    msg_.periodic = std::move(arg);
    return Init_MovePeriodic_Request_acc(msg_);
  }

private:
  ::dsr_msgs2::srv::MovePeriodic_Request msg_;
};

class Init_MovePeriodic_Request_amp
{
public:
  Init_MovePeriodic_Request_amp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MovePeriodic_Request_periodic amp(::dsr_msgs2::srv::MovePeriodic_Request::_amp_type arg)
  {
    msg_.amp = std::move(arg);
    return Init_MovePeriodic_Request_periodic(msg_);
  }

private:
  ::dsr_msgs2::srv::MovePeriodic_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MovePeriodic_Request>()
{
  return dsr_msgs2::srv::builder::Init_MovePeriodic_Request_amp();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MovePeriodic_Response_success
{
public:
  Init_MovePeriodic_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::MovePeriodic_Response success(::dsr_msgs2::srv::MovePeriodic_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MovePeriodic_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MovePeriodic_Response>()
{
  return dsr_msgs2::srv::builder::Init_MovePeriodic_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_PERIODIC__BUILDER_HPP_
