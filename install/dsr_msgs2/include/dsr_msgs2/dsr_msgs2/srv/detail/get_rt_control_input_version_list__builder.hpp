// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetRtControlInputVersionList.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_RT_CONTROL_INPUT_VERSION_LIST__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_RT_CONTROL_INPUT_VERSION_LIST__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_rt_control_input_version_list__struct.hpp"
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
auto build<::dsr_msgs2::srv::GetRtControlInputVersionList_Request>()
{
  return ::dsr_msgs2::srv::GetRtControlInputVersionList_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetRtControlInputVersionList_Response_version
{
public:
  explicit Init_GetRtControlInputVersionList_Response_version(::dsr_msgs2::srv::GetRtControlInputVersionList_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetRtControlInputVersionList_Response version(::dsr_msgs2::srv::GetRtControlInputVersionList_Response::_version_type arg)
  {
    msg_.version = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetRtControlInputVersionList_Response msg_;
};

class Init_GetRtControlInputVersionList_Response_success
{
public:
  Init_GetRtControlInputVersionList_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetRtControlInputVersionList_Response_version success(::dsr_msgs2::srv::GetRtControlInputVersionList_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GetRtControlInputVersionList_Response_version(msg_);
  }

private:
  ::dsr_msgs2::srv::GetRtControlInputVersionList_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetRtControlInputVersionList_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetRtControlInputVersionList_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_RT_CONTROL_INPUT_VERSION_LIST__BUILDER_HPP_
