// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/IsDoneBoltTightening.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__IS_DONE_BOLT_TIGHTENING__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__IS_DONE_BOLT_TIGHTENING__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/is_done_bolt_tightening__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_IsDoneBoltTightening_Request_axis
{
public:
  explicit Init_IsDoneBoltTightening_Request_axis(::dsr_msgs2::srv::IsDoneBoltTightening_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::IsDoneBoltTightening_Request axis(::dsr_msgs2::srv::IsDoneBoltTightening_Request::_axis_type arg)
  {
    msg_.axis = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::IsDoneBoltTightening_Request msg_;
};

class Init_IsDoneBoltTightening_Request_timeout
{
public:
  explicit Init_IsDoneBoltTightening_Request_timeout(::dsr_msgs2::srv::IsDoneBoltTightening_Request & msg)
  : msg_(msg)
  {}
  Init_IsDoneBoltTightening_Request_axis timeout(::dsr_msgs2::srv::IsDoneBoltTightening_Request::_timeout_type arg)
  {
    msg_.timeout = std::move(arg);
    return Init_IsDoneBoltTightening_Request_axis(msg_);
  }

private:
  ::dsr_msgs2::srv::IsDoneBoltTightening_Request msg_;
};

class Init_IsDoneBoltTightening_Request_m
{
public:
  Init_IsDoneBoltTightening_Request_m()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_IsDoneBoltTightening_Request_timeout m(::dsr_msgs2::srv::IsDoneBoltTightening_Request::_m_type arg)
  {
    msg_.m = std::move(arg);
    return Init_IsDoneBoltTightening_Request_timeout(msg_);
  }

private:
  ::dsr_msgs2::srv::IsDoneBoltTightening_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::IsDoneBoltTightening_Request>()
{
  return dsr_msgs2::srv::builder::Init_IsDoneBoltTightening_Request_m();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_IsDoneBoltTightening_Response_success
{
public:
  Init_IsDoneBoltTightening_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::IsDoneBoltTightening_Response success(::dsr_msgs2::srv::IsDoneBoltTightening_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::IsDoneBoltTightening_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::IsDoneBoltTightening_Response>()
{
  return dsr_msgs2::srv::builder::Init_IsDoneBoltTightening_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__IS_DONE_BOLT_TIGHTENING__BUILDER_HPP_
