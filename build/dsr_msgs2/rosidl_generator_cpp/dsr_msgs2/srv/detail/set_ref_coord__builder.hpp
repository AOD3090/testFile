// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetRefCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_REF_COORD__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_REF_COORD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_ref_coord__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRefCoord_Request_coord
{
public:
  Init_SetRefCoord_Request_coord()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRefCoord_Request coord(::dsr_msgs2::srv::SetRefCoord_Request::_coord_type arg)
  {
    msg_.coord = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRefCoord_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRefCoord_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetRefCoord_Request_coord();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRefCoord_Response_success
{
public:
  Init_SetRefCoord_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRefCoord_Response success(::dsr_msgs2::srv::SetRefCoord_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRefCoord_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRefCoord_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetRefCoord_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_REF_COORD__BUILDER_HPP_
