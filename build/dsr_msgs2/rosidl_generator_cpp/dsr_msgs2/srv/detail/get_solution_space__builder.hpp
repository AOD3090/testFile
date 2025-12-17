// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetSolutionSpace.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_SOLUTION_SPACE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_SOLUTION_SPACE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_solution_space__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetSolutionSpace_Request_pos
{
public:
  Init_GetSolutionSpace_Request_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetSolutionSpace_Request pos(::dsr_msgs2::srv::GetSolutionSpace_Request::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetSolutionSpace_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetSolutionSpace_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetSolutionSpace_Request_pos();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetSolutionSpace_Response_success
{
public:
  explicit Init_GetSolutionSpace_Response_success(::dsr_msgs2::srv::GetSolutionSpace_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetSolutionSpace_Response success(::dsr_msgs2::srv::GetSolutionSpace_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetSolutionSpace_Response msg_;
};

class Init_GetSolutionSpace_Response_sol_space
{
public:
  Init_GetSolutionSpace_Response_sol_space()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetSolutionSpace_Response_success sol_space(::dsr_msgs2::srv::GetSolutionSpace_Response::_sol_space_type arg)
  {
    msg_.sol_space = std::move(arg);
    return Init_GetSolutionSpace_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetSolutionSpace_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetSolutionSpace_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetSolutionSpace_Response_sol_space();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_SOLUTION_SPACE__BUILDER_HPP_
