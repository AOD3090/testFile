// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetRtControlOutputDataList.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_RT_CONTROL_OUTPUT_DATA_LIST__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_RT_CONTROL_OUTPUT_DATA_LIST__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_rt_control_output_data_list__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetRtControlOutputDataList_Request_version
{
public:
  Init_GetRtControlOutputDataList_Request_version()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetRtControlOutputDataList_Request version(::dsr_msgs2::srv::GetRtControlOutputDataList_Request::_version_type arg)
  {
    msg_.version = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetRtControlOutputDataList_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetRtControlOutputDataList_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetRtControlOutputDataList_Request_version();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetRtControlOutputDataList_Response_data
{
public:
  explicit Init_GetRtControlOutputDataList_Response_data(::dsr_msgs2::srv::GetRtControlOutputDataList_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetRtControlOutputDataList_Response data(::dsr_msgs2::srv::GetRtControlOutputDataList_Response::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetRtControlOutputDataList_Response msg_;
};

class Init_GetRtControlOutputDataList_Response_success
{
public:
  Init_GetRtControlOutputDataList_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetRtControlOutputDataList_Response_data success(::dsr_msgs2::srv::GetRtControlOutputDataList_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GetRtControlOutputDataList_Response_data(msg_);
  }

private:
  ::dsr_msgs2::srv::GetRtControlOutputDataList_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetRtControlOutputDataList_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetRtControlOutputDataList_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_RT_CONTROL_OUTPUT_DATA_LIST__BUILDER_HPP_
