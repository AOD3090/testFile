// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/GetToolForce.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_TOOL_FORCE__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_TOOL_FORCE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__GetToolForce_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__GetToolForce_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetToolForce_Request_
{
  using Type = GetToolForce_Request_<ContainerAllocator>;

  explicit GetToolForce_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->ref = 0;
    }
  }

  explicit GetToolForce_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->ref = 0;
    }
  }

  // field types and members
  using _ref_type =
    int8_t;
  _ref_type ref;

  // setters for named parameter idiom
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__GetToolForce_Request
    std::shared_ptr<dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__GetToolForce_Request
    std::shared_ptr<dsr_msgs2::srv::GetToolForce_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetToolForce_Request_ & other) const
  {
    if (this->ref != other.ref) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetToolForce_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetToolForce_Request_

// alias to use template instance with default allocator
using GetToolForce_Request =
  dsr_msgs2::srv::GetToolForce_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__GetToolForce_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__GetToolForce_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetToolForce_Response_
{
  using Type = GetToolForce_Response_<ContainerAllocator>;

  explicit GetToolForce_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->tool_force.begin(), this->tool_force.end(), 0.0);
      this->success = false;
    }
  }

  explicit GetToolForce_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : tool_force(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->tool_force.begin(), this->tool_force.end(), 0.0);
      this->success = false;
    }
  }

  // field types and members
  using _tool_force_type =
    std::array<double, 6>;
  _tool_force_type tool_force;
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__tool_force(
    const std::array<double, 6> & _arg)
  {
    this->tool_force = _arg;
    return *this;
  }
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__GetToolForce_Response
    std::shared_ptr<dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__GetToolForce_Response
    std::shared_ptr<dsr_msgs2::srv::GetToolForce_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetToolForce_Response_ & other) const
  {
    if (this->tool_force != other.tool_force) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetToolForce_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetToolForce_Response_

// alias to use template instance with default allocator
using GetToolForce_Response =
  dsr_msgs2::srv::GetToolForce_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct GetToolForce
{
  using Request = dsr_msgs2::srv::GetToolForce_Request;
  using Response = dsr_msgs2::srv::GetToolForce_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_TOOL_FORCE__STRUCT_HPP_
