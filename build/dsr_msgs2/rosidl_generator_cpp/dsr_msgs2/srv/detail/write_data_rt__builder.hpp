// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/WriteDataRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__WRITE_DATA_RT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__WRITE_DATA_RT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/write_data_rt__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_WriteDataRt_Request_external_analog_output
{
public:
  explicit Init_WriteDataRt_Request_external_analog_output(::dsr_msgs2::srv::WriteDataRt_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::WriteDataRt_Request external_analog_output(::dsr_msgs2::srv::WriteDataRt_Request::_external_analog_output_type arg)
  {
    msg_.external_analog_output = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::WriteDataRt_Request msg_;
};

class Init_WriteDataRt_Request_external_analog_input
{
public:
  explicit Init_WriteDataRt_Request_external_analog_input(::dsr_msgs2::srv::WriteDataRt_Request & msg)
  : msg_(msg)
  {}
  Init_WriteDataRt_Request_external_analog_output external_analog_input(::dsr_msgs2::srv::WriteDataRt_Request::_external_analog_input_type arg)
  {
    msg_.external_analog_input = std::move(arg);
    return Init_WriteDataRt_Request_external_analog_output(msg_);
  }

private:
  ::dsr_msgs2::srv::WriteDataRt_Request msg_;
};

class Init_WriteDataRt_Request_external_digital_output
{
public:
  explicit Init_WriteDataRt_Request_external_digital_output(::dsr_msgs2::srv::WriteDataRt_Request & msg)
  : msg_(msg)
  {}
  Init_WriteDataRt_Request_external_analog_input external_digital_output(::dsr_msgs2::srv::WriteDataRt_Request::_external_digital_output_type arg)
  {
    msg_.external_digital_output = std::move(arg);
    return Init_WriteDataRt_Request_external_analog_input(msg_);
  }

private:
  ::dsr_msgs2::srv::WriteDataRt_Request msg_;
};

class Init_WriteDataRt_Request_external_digital_input
{
public:
  explicit Init_WriteDataRt_Request_external_digital_input(::dsr_msgs2::srv::WriteDataRt_Request & msg)
  : msg_(msg)
  {}
  Init_WriteDataRt_Request_external_digital_output external_digital_input(::dsr_msgs2::srv::WriteDataRt_Request::_external_digital_input_type arg)
  {
    msg_.external_digital_input = std::move(arg);
    return Init_WriteDataRt_Request_external_digital_output(msg_);
  }

private:
  ::dsr_msgs2::srv::WriteDataRt_Request msg_;
};

class Init_WriteDataRt_Request_external_force_torque
{
public:
  Init_WriteDataRt_Request_external_force_torque()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_WriteDataRt_Request_external_digital_input external_force_torque(::dsr_msgs2::srv::WriteDataRt_Request::_external_force_torque_type arg)
  {
    msg_.external_force_torque = std::move(arg);
    return Init_WriteDataRt_Request_external_digital_input(msg_);
  }

private:
  ::dsr_msgs2::srv::WriteDataRt_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::WriteDataRt_Request>()
{
  return dsr_msgs2::srv::builder::Init_WriteDataRt_Request_external_force_torque();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_WriteDataRt_Response_success
{
public:
  Init_WriteDataRt_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::WriteDataRt_Response success(::dsr_msgs2::srv::WriteDataRt_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::WriteDataRt_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::WriteDataRt_Response>()
{
  return dsr_msgs2::srv::builder::Init_WriteDataRt_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__WRITE_DATA_RT__BUILDER_HPP_
