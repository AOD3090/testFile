// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/ReadDataRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/read_data_rt__struct.hpp"
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
auto build<::dsr_msgs2::srv::ReadDataRt_Request>()
{
  return ::dsr_msgs2::srv::ReadDataRt_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ReadDataRt_Response_data
{
public:
  Init_ReadDataRt_Response_data()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ReadDataRt_Response data(::dsr_msgs2::srv::ReadDataRt_Response::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ReadDataRt_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ReadDataRt_Response>()
{
  return dsr_msgs2::srv::builder::Init_ReadDataRt_Response_data();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__BUILDER_HPP_
