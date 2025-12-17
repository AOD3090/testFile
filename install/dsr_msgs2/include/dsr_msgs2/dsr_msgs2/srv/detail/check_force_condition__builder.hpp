// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/CheckForceCondition.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CHECK_FORCE_CONDITION__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__CHECK_FORCE_CONDITION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/check_force_condition__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_CheckForceCondition_Request_ref
{
public:
  explicit Init_CheckForceCondition_Request_ref(::dsr_msgs2::srv::CheckForceCondition_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::CheckForceCondition_Request ref(::dsr_msgs2::srv::CheckForceCondition_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckForceCondition_Request msg_;
};

class Init_CheckForceCondition_Request_max
{
public:
  explicit Init_CheckForceCondition_Request_max(::dsr_msgs2::srv::CheckForceCondition_Request & msg)
  : msg_(msg)
  {}
  Init_CheckForceCondition_Request_ref max(::dsr_msgs2::srv::CheckForceCondition_Request::_max_type arg)
  {
    msg_.max = std::move(arg);
    return Init_CheckForceCondition_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckForceCondition_Request msg_;
};

class Init_CheckForceCondition_Request_min
{
public:
  explicit Init_CheckForceCondition_Request_min(::dsr_msgs2::srv::CheckForceCondition_Request & msg)
  : msg_(msg)
  {}
  Init_CheckForceCondition_Request_max min(::dsr_msgs2::srv::CheckForceCondition_Request::_min_type arg)
  {
    msg_.min = std::move(arg);
    return Init_CheckForceCondition_Request_max(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckForceCondition_Request msg_;
};

class Init_CheckForceCondition_Request_axis
{
public:
  Init_CheckForceCondition_Request_axis()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CheckForceCondition_Request_min axis(::dsr_msgs2::srv::CheckForceCondition_Request::_axis_type arg)
  {
    msg_.axis = std::move(arg);
    return Init_CheckForceCondition_Request_min(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckForceCondition_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::CheckForceCondition_Request>()
{
  return dsr_msgs2::srv::builder::Init_CheckForceCondition_Request_axis();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_CheckForceCondition_Response_success
{
public:
  Init_CheckForceCondition_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::CheckForceCondition_Response success(::dsr_msgs2::srv::CheckForceCondition_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckForceCondition_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::CheckForceCondition_Response>()
{
  return dsr_msgs2::srv::builder::Init_CheckForceCondition_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CHECK_FORCE_CONDITION__BUILDER_HPP_
