// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetUserCartCoord3.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_user_cart_coord3__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetUserCartCoord3_Request_ref
{
public:
  explicit Init_SetUserCartCoord3_Request_ref(::dsr_msgs2::srv::SetUserCartCoord3_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetUserCartCoord3_Request ref(::dsr_msgs2::srv::SetUserCartCoord3_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetUserCartCoord3_Request msg_;
};

class Init_SetUserCartCoord3_Request_pos
{
public:
  explicit Init_SetUserCartCoord3_Request_pos(::dsr_msgs2::srv::SetUserCartCoord3_Request & msg)
  : msg_(msg)
  {}
  Init_SetUserCartCoord3_Request_ref pos(::dsr_msgs2::srv::SetUserCartCoord3_Request::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_SetUserCartCoord3_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::SetUserCartCoord3_Request msg_;
};

class Init_SetUserCartCoord3_Request_v1
{
public:
  explicit Init_SetUserCartCoord3_Request_v1(::dsr_msgs2::srv::SetUserCartCoord3_Request & msg)
  : msg_(msg)
  {}
  Init_SetUserCartCoord3_Request_pos v1(::dsr_msgs2::srv::SetUserCartCoord3_Request::_v1_type arg)
  {
    msg_.v1 = std::move(arg);
    return Init_SetUserCartCoord3_Request_pos(msg_);
  }

private:
  ::dsr_msgs2::srv::SetUserCartCoord3_Request msg_;
};

class Init_SetUserCartCoord3_Request_u1
{
public:
  Init_SetUserCartCoord3_Request_u1()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetUserCartCoord3_Request_v1 u1(::dsr_msgs2::srv::SetUserCartCoord3_Request::_u1_type arg)
  {
    msg_.u1 = std::move(arg);
    return Init_SetUserCartCoord3_Request_v1(msg_);
  }

private:
  ::dsr_msgs2::srv::SetUserCartCoord3_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetUserCartCoord3_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetUserCartCoord3_Request_u1();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetUserCartCoord3_Response_success
{
public:
  explicit Init_SetUserCartCoord3_Response_success(::dsr_msgs2::srv::SetUserCartCoord3_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetUserCartCoord3_Response success(::dsr_msgs2::srv::SetUserCartCoord3_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetUserCartCoord3_Response msg_;
};

class Init_SetUserCartCoord3_Response_id
{
public:
  Init_SetUserCartCoord3_Response_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetUserCartCoord3_Response_success id(::dsr_msgs2::srv::SetUserCartCoord3_Response::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_SetUserCartCoord3_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::SetUserCartCoord3_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetUserCartCoord3_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetUserCartCoord3_Response_id();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_USER_CART_COORD3__BUILDER_HPP_
