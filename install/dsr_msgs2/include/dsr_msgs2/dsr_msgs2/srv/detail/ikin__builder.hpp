// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/Ikin.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__IKIN__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__IKIN__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/ikin__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_Ikin_Request_ref
{
public:
  explicit Init_Ikin_Request_ref(::dsr_msgs2::srv::Ikin_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::Ikin_Request ref(::dsr_msgs2::srv::Ikin_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::Ikin_Request msg_;
};

class Init_Ikin_Request_sol_space
{
public:
  explicit Init_Ikin_Request_sol_space(::dsr_msgs2::srv::Ikin_Request & msg)
  : msg_(msg)
  {}
  Init_Ikin_Request_ref sol_space(::dsr_msgs2::srv::Ikin_Request::_sol_space_type arg)
  {
    msg_.sol_space = std::move(arg);
    return Init_Ikin_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::Ikin_Request msg_;
};

class Init_Ikin_Request_pos
{
public:
  Init_Ikin_Request_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Ikin_Request_sol_space pos(::dsr_msgs2::srv::Ikin_Request::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_Ikin_Request_sol_space(msg_);
  }

private:
  ::dsr_msgs2::srv::Ikin_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::Ikin_Request>()
{
  return dsr_msgs2::srv::builder::Init_Ikin_Request_pos();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_Ikin_Response_success
{
public:
  explicit Init_Ikin_Response_success(::dsr_msgs2::srv::Ikin_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::Ikin_Response success(::dsr_msgs2::srv::Ikin_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::Ikin_Response msg_;
};

class Init_Ikin_Response_conv_posj
{
public:
  Init_Ikin_Response_conv_posj()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Ikin_Response_success conv_posj(::dsr_msgs2::srv::Ikin_Response::_conv_posj_type arg)
  {
    msg_.conv_posj = std::move(arg);
    return Init_Ikin_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::Ikin_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::Ikin_Response>()
{
  return dsr_msgs2::srv::builder::Init_Ikin_Response_conv_posj();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__IKIN__BUILDER_HPP_
