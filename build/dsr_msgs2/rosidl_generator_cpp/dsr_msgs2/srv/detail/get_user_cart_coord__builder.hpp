// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetUserCartCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_user_cart_coord__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetUserCartCoord_Request_id
{
public:
  Init_GetUserCartCoord_Request_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetUserCartCoord_Request id(::dsr_msgs2::srv::GetUserCartCoord_Request::_id_type arg)
  {
    msg_.id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetUserCartCoord_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetUserCartCoord_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetUserCartCoord_Request_id();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetUserCartCoord_Response_success
{
public:
  explicit Init_GetUserCartCoord_Response_success(::dsr_msgs2::srv::GetUserCartCoord_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetUserCartCoord_Response success(::dsr_msgs2::srv::GetUserCartCoord_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetUserCartCoord_Response msg_;
};

class Init_GetUserCartCoord_Response_ref
{
public:
  explicit Init_GetUserCartCoord_Response_ref(::dsr_msgs2::srv::GetUserCartCoord_Response & msg)
  : msg_(msg)
  {}
  Init_GetUserCartCoord_Response_success ref(::dsr_msgs2::srv::GetUserCartCoord_Response::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_GetUserCartCoord_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetUserCartCoord_Response msg_;
};

class Init_GetUserCartCoord_Response_conv_posx
{
public:
  Init_GetUserCartCoord_Response_conv_posx()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetUserCartCoord_Response_ref conv_posx(::dsr_msgs2::srv::GetUserCartCoord_Response::_conv_posx_type arg)
  {
    msg_.conv_posx = std::move(arg);
    return Init_GetUserCartCoord_Response_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::GetUserCartCoord_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetUserCartCoord_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetUserCartCoord_Response_conv_posx();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_USER_CART_COORD__BUILDER_HPP_
