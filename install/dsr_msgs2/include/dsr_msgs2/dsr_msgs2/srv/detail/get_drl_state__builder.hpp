// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetDrlState.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_DRL_STATE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_DRL_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_drl_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetDrlState_Request>()
{
  return ::dsr_msgs2::srv::GetDrlState_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetDrlState_Response_success
{
public:
  explicit Init_GetDrlState_Response_success(::dsr_msgs2::srv::GetDrlState_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetDrlState_Response success(::dsr_msgs2::srv::GetDrlState_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetDrlState_Response msg_;
};

class Init_GetDrlState_Response_drl_state
{
public:
  Init_GetDrlState_Response_drl_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetDrlState_Response_success drl_state(::dsr_msgs2::srv::GetDrlState_Response::_drl_state_type arg)
  {
    msg_.drl_state = std::move(arg);
    return Init_GetDrlState_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetDrlState_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetDrlState_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetDrlState_Response_drl_state();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_DRL_STATE__BUILDER_HPP_
