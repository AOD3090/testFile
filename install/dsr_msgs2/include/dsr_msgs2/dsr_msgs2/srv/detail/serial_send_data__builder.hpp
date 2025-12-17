// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SerialSendData.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SERIAL_SEND_DATA__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SERIAL_SEND_DATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/serial_send_data__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SerialSendData_Request_data
{
public:
  Init_SerialSendData_Request_data()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SerialSendData_Request data(::dsr_msgs2::srv::SerialSendData_Request::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SerialSendData_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SerialSendData_Request>()
{
  return dsr_msgs2::srv::builder::Init_SerialSendData_Request_data();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SerialSendData_Response_success
{
public:
  Init_SerialSendData_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SerialSendData_Response success(::dsr_msgs2::srv::SerialSendData_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SerialSendData_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SerialSendData_Response>()
{
  return dsr_msgs2::srv::builder::Init_SerialSendData_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SERIAL_SEND_DATA__BUILDER_HPP_
