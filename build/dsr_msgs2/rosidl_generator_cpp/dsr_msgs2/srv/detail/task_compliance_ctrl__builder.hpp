// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/TaskComplianceCtrl.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__TASK_COMPLIANCE_CTRL__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__TASK_COMPLIANCE_CTRL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/task_compliance_ctrl__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_TaskComplianceCtrl_Request_time
{
public:
  explicit Init_TaskComplianceCtrl_Request_time(::dsr_msgs2::srv::TaskComplianceCtrl_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::TaskComplianceCtrl_Request time(::dsr_msgs2::srv::TaskComplianceCtrl_Request::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::TaskComplianceCtrl_Request msg_;
};

class Init_TaskComplianceCtrl_Request_ref
{
public:
  explicit Init_TaskComplianceCtrl_Request_ref(::dsr_msgs2::srv::TaskComplianceCtrl_Request & msg)
  : msg_(msg)
  {}
  Init_TaskComplianceCtrl_Request_time ref(::dsr_msgs2::srv::TaskComplianceCtrl_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_TaskComplianceCtrl_Request_time(msg_);
  }

private:
  ::dsr_msgs2::srv::TaskComplianceCtrl_Request msg_;
};

class Init_TaskComplianceCtrl_Request_stx
{
public:
  Init_TaskComplianceCtrl_Request_stx()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TaskComplianceCtrl_Request_ref stx(::dsr_msgs2::srv::TaskComplianceCtrl_Request::_stx_type arg)
  {
    msg_.stx = std::move(arg);
    return Init_TaskComplianceCtrl_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::TaskComplianceCtrl_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::TaskComplianceCtrl_Request>()
{
  return dsr_msgs2::srv::builder::Init_TaskComplianceCtrl_Request_stx();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_TaskComplianceCtrl_Response_success
{
public:
  Init_TaskComplianceCtrl_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::TaskComplianceCtrl_Response success(::dsr_msgs2::srv::TaskComplianceCtrl_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::TaskComplianceCtrl_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::TaskComplianceCtrl_Response>()
{
  return dsr_msgs2::srv::builder::Init_TaskComplianceCtrl_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__TASK_COMPLIANCE_CTRL__BUILDER_HPP_
