// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/AlignAxis1.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__ALIGN_AXIS1__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__ALIGN_AXIS1__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/align_axis1__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_AlignAxis1_Request_ref
{
public:
  explicit Init_AlignAxis1_Request_ref(::dsr_msgs2::srv::AlignAxis1_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::AlignAxis1_Request ref(::dsr_msgs2::srv::AlignAxis1_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::AlignAxis1_Request msg_;
};

class Init_AlignAxis1_Request_axis
{
public:
  explicit Init_AlignAxis1_Request_axis(::dsr_msgs2::srv::AlignAxis1_Request & msg)
  : msg_(msg)
  {}
  Init_AlignAxis1_Request_ref axis(::dsr_msgs2::srv::AlignAxis1_Request::_axis_type arg)
  {
    msg_.axis = std::move(arg);
    return Init_AlignAxis1_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::AlignAxis1_Request msg_;
};

class Init_AlignAxis1_Request_source_vect
{
public:
  explicit Init_AlignAxis1_Request_source_vect(::dsr_msgs2::srv::AlignAxis1_Request & msg)
  : msg_(msg)
  {}
  Init_AlignAxis1_Request_axis source_vect(::dsr_msgs2::srv::AlignAxis1_Request::_source_vect_type arg)
  {
    msg_.source_vect = std::move(arg);
    return Init_AlignAxis1_Request_axis(msg_);
  }

private:
  ::dsr_msgs2::srv::AlignAxis1_Request msg_;
};

class Init_AlignAxis1_Request_x3
{
public:
  explicit Init_AlignAxis1_Request_x3(::dsr_msgs2::srv::AlignAxis1_Request & msg)
  : msg_(msg)
  {}
  Init_AlignAxis1_Request_source_vect x3(::dsr_msgs2::srv::AlignAxis1_Request::_x3_type arg)
  {
    msg_.x3 = std::move(arg);
    return Init_AlignAxis1_Request_source_vect(msg_);
  }

private:
  ::dsr_msgs2::srv::AlignAxis1_Request msg_;
};

class Init_AlignAxis1_Request_x2
{
public:
  explicit Init_AlignAxis1_Request_x2(::dsr_msgs2::srv::AlignAxis1_Request & msg)
  : msg_(msg)
  {}
  Init_AlignAxis1_Request_x3 x2(::dsr_msgs2::srv::AlignAxis1_Request::_x2_type arg)
  {
    msg_.x2 = std::move(arg);
    return Init_AlignAxis1_Request_x3(msg_);
  }

private:
  ::dsr_msgs2::srv::AlignAxis1_Request msg_;
};

class Init_AlignAxis1_Request_x1
{
public:
  Init_AlignAxis1_Request_x1()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_AlignAxis1_Request_x2 x1(::dsr_msgs2::srv::AlignAxis1_Request::_x1_type arg)
  {
    msg_.x1 = std::move(arg);
    return Init_AlignAxis1_Request_x2(msg_);
  }

private:
  ::dsr_msgs2::srv::AlignAxis1_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::AlignAxis1_Request>()
{
  return dsr_msgs2::srv::builder::Init_AlignAxis1_Request_x1();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_AlignAxis1_Response_success
{
public:
  Init_AlignAxis1_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::AlignAxis1_Response success(::dsr_msgs2::srv::AlignAxis1_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::AlignAxis1_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::AlignAxis1_Response>()
{
  return dsr_msgs2::srv::builder::Init_AlignAxis1_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__ALIGN_AXIS1__BUILDER_HPP_
