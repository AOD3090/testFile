// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetWorkpieceWeight.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_WORKPIECE_WEIGHT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_WORKPIECE_WEIGHT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_workpiece_weight__struct.hpp"
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
auto build<::dsr_msgs2::srv::GetWorkpieceWeight_Request>()
{
  return ::dsr_msgs2::srv::GetWorkpieceWeight_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetWorkpieceWeight_Response_success
{
public:
  explicit Init_GetWorkpieceWeight_Response_success(::dsr_msgs2::srv::GetWorkpieceWeight_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetWorkpieceWeight_Response success(::dsr_msgs2::srv::GetWorkpieceWeight_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetWorkpieceWeight_Response msg_;
};

class Init_GetWorkpieceWeight_Response_weight
{
public:
  Init_GetWorkpieceWeight_Response_weight()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetWorkpieceWeight_Response_success weight(::dsr_msgs2::srv::GetWorkpieceWeight_Response::_weight_type arg)
  {
    msg_.weight = std::move(arg);
    return Init_GetWorkpieceWeight_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetWorkpieceWeight_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetWorkpieceWeight_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetWorkpieceWeight_Response_weight();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_WORKPIECE_WEIGHT__BUILDER_HPP_
