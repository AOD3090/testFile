// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/CheckPositionCondition.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CHECK_POSITION_CONDITION__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__CHECK_POSITION_CONDITION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/check_position_condition__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_CheckPositionCondition_Request_pos
{
public:
  explicit Init_CheckPositionCondition_Request_pos(::dsr_msgs2::srv::CheckPositionCondition_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::CheckPositionCondition_Request pos(::dsr_msgs2::srv::CheckPositionCondition_Request::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckPositionCondition_Request msg_;
};

class Init_CheckPositionCondition_Request_mode
{
public:
  explicit Init_CheckPositionCondition_Request_mode(::dsr_msgs2::srv::CheckPositionCondition_Request & msg)
  : msg_(msg)
  {}
  Init_CheckPositionCondition_Request_pos mode(::dsr_msgs2::srv::CheckPositionCondition_Request::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_CheckPositionCondition_Request_pos(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckPositionCondition_Request msg_;
};

class Init_CheckPositionCondition_Request_ref
{
public:
  explicit Init_CheckPositionCondition_Request_ref(::dsr_msgs2::srv::CheckPositionCondition_Request & msg)
  : msg_(msg)
  {}
  Init_CheckPositionCondition_Request_mode ref(::dsr_msgs2::srv::CheckPositionCondition_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_CheckPositionCondition_Request_mode(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckPositionCondition_Request msg_;
};

class Init_CheckPositionCondition_Request_max
{
public:
  explicit Init_CheckPositionCondition_Request_max(::dsr_msgs2::srv::CheckPositionCondition_Request & msg)
  : msg_(msg)
  {}
  Init_CheckPositionCondition_Request_ref max(::dsr_msgs2::srv::CheckPositionCondition_Request::_max_type arg)
  {
    msg_.max = std::move(arg);
    return Init_CheckPositionCondition_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckPositionCondition_Request msg_;
};

class Init_CheckPositionCondition_Request_min
{
public:
  explicit Init_CheckPositionCondition_Request_min(::dsr_msgs2::srv::CheckPositionCondition_Request & msg)
  : msg_(msg)
  {}
  Init_CheckPositionCondition_Request_max min(::dsr_msgs2::srv::CheckPositionCondition_Request::_min_type arg)
  {
    msg_.min = std::move(arg);
    return Init_CheckPositionCondition_Request_max(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckPositionCondition_Request msg_;
};

class Init_CheckPositionCondition_Request_axis
{
public:
  Init_CheckPositionCondition_Request_axis()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CheckPositionCondition_Request_min axis(::dsr_msgs2::srv::CheckPositionCondition_Request::_axis_type arg)
  {
    msg_.axis = std::move(arg);
    return Init_CheckPositionCondition_Request_min(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckPositionCondition_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::CheckPositionCondition_Request>()
{
  return dsr_msgs2::srv::builder::Init_CheckPositionCondition_Request_axis();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_CheckPositionCondition_Response_success
{
public:
  Init_CheckPositionCondition_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::CheckPositionCondition_Response success(::dsr_msgs2::srv::CheckPositionCondition_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::CheckPositionCondition_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::CheckPositionCondition_Response>()
{
  return dsr_msgs2::srv::builder::Init_CheckPositionCondition_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CHECK_POSITION_CONDITION__BUILDER_HPP_
