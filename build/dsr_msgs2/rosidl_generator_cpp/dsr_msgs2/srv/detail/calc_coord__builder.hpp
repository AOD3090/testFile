// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/CalcCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CALC_COORD__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__CALC_COORD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/calc_coord__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_CalcCoord_Request_mod
{
public:
  explicit Init_CalcCoord_Request_mod(::dsr_msgs2::srv::CalcCoord_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::CalcCoord_Request mod(::dsr_msgs2::srv::CalcCoord_Request::_mod_type arg)
  {
    msg_.mod = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Request msg_;
};

class Init_CalcCoord_Request_ref
{
public:
  explicit Init_CalcCoord_Request_ref(::dsr_msgs2::srv::CalcCoord_Request & msg)
  : msg_(msg)
  {}
  Init_CalcCoord_Request_mod ref(::dsr_msgs2::srv::CalcCoord_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_CalcCoord_Request_mod(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Request msg_;
};

class Init_CalcCoord_Request_x4
{
public:
  explicit Init_CalcCoord_Request_x4(::dsr_msgs2::srv::CalcCoord_Request & msg)
  : msg_(msg)
  {}
  Init_CalcCoord_Request_ref x4(::dsr_msgs2::srv::CalcCoord_Request::_x4_type arg)
  {
    msg_.x4 = std::move(arg);
    return Init_CalcCoord_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Request msg_;
};

class Init_CalcCoord_Request_x3
{
public:
  explicit Init_CalcCoord_Request_x3(::dsr_msgs2::srv::CalcCoord_Request & msg)
  : msg_(msg)
  {}
  Init_CalcCoord_Request_x4 x3(::dsr_msgs2::srv::CalcCoord_Request::_x3_type arg)
  {
    msg_.x3 = std::move(arg);
    return Init_CalcCoord_Request_x4(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Request msg_;
};

class Init_CalcCoord_Request_x2
{
public:
  explicit Init_CalcCoord_Request_x2(::dsr_msgs2::srv::CalcCoord_Request & msg)
  : msg_(msg)
  {}
  Init_CalcCoord_Request_x3 x2(::dsr_msgs2::srv::CalcCoord_Request::_x2_type arg)
  {
    msg_.x2 = std::move(arg);
    return Init_CalcCoord_Request_x3(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Request msg_;
};

class Init_CalcCoord_Request_x1
{
public:
  explicit Init_CalcCoord_Request_x1(::dsr_msgs2::srv::CalcCoord_Request & msg)
  : msg_(msg)
  {}
  Init_CalcCoord_Request_x2 x1(::dsr_msgs2::srv::CalcCoord_Request::_x1_type arg)
  {
    msg_.x1 = std::move(arg);
    return Init_CalcCoord_Request_x2(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Request msg_;
};

class Init_CalcCoord_Request_input_pos_cnt
{
public:
  Init_CalcCoord_Request_input_pos_cnt()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CalcCoord_Request_x1 input_pos_cnt(::dsr_msgs2::srv::CalcCoord_Request::_input_pos_cnt_type arg)
  {
    msg_.input_pos_cnt = std::move(arg);
    return Init_CalcCoord_Request_x1(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::CalcCoord_Request>()
{
  return dsr_msgs2::srv::builder::Init_CalcCoord_Request_input_pos_cnt();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_CalcCoord_Response_success
{
public:
  explicit Init_CalcCoord_Response_success(::dsr_msgs2::srv::CalcCoord_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::CalcCoord_Response success(::dsr_msgs2::srv::CalcCoord_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Response msg_;
};

class Init_CalcCoord_Response_conv_posx
{
public:
  Init_CalcCoord_Response_conv_posx()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CalcCoord_Response_success conv_posx(::dsr_msgs2::srv::CalcCoord_Response::_conv_posx_type arg)
  {
    msg_.conv_posx = std::move(arg);
    return Init_CalcCoord_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::CalcCoord_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::CalcCoord_Response>()
{
  return dsr_msgs2::srv::builder::Init_CalcCoord_Response_conv_posx();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CALC_COORD__BUILDER_HPP_
