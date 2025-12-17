// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/ParallelAxis2.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__PARALLEL_AXIS2__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__PARALLEL_AXIS2__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/parallel_axis2__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ParallelAxis2_Request_ref
{
public:
  explicit Init_ParallelAxis2_Request_ref(::dsr_msgs2::srv::ParallelAxis2_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::ParallelAxis2_Request ref(::dsr_msgs2::srv::ParallelAxis2_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ParallelAxis2_Request msg_;
};

class Init_ParallelAxis2_Request_axis
{
public:
  explicit Init_ParallelAxis2_Request_axis(::dsr_msgs2::srv::ParallelAxis2_Request & msg)
  : msg_(msg)
  {}
  Init_ParallelAxis2_Request_ref axis(::dsr_msgs2::srv::ParallelAxis2_Request::_axis_type arg)
  {
    msg_.axis = std::move(arg);
    return Init_ParallelAxis2_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::ParallelAxis2_Request msg_;
};

class Init_ParallelAxis2_Request_vect
{
public:
  Init_ParallelAxis2_Request_vect()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ParallelAxis2_Request_axis vect(::dsr_msgs2::srv::ParallelAxis2_Request::_vect_type arg)
  {
    msg_.vect = std::move(arg);
    return Init_ParallelAxis2_Request_axis(msg_);
  }

private:
  ::dsr_msgs2::srv::ParallelAxis2_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ParallelAxis2_Request>()
{
  return dsr_msgs2::srv::builder::Init_ParallelAxis2_Request_vect();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ParallelAxis2_Response_success
{
public:
  Init_ParallelAxis2_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ParallelAxis2_Response success(::dsr_msgs2::srv::ParallelAxis2_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ParallelAxis2_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ParallelAxis2_Response>()
{
  return dsr_msgs2::srv::builder::Init_ParallelAxis2_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__PARALLEL_AXIS2__BUILDER_HPP_
