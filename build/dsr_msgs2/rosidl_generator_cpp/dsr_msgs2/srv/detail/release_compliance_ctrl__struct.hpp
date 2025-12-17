// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/ReleaseComplianceCtrl.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__RELEASE_COMPLIANCE_CTRL__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__RELEASE_COMPLIANCE_CTRL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__ReleaseComplianceCtrl_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__ReleaseComplianceCtrl_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ReleaseComplianceCtrl_Request_
{
  using Type = ReleaseComplianceCtrl_Request_<ContainerAllocator>;

  explicit ReleaseComplianceCtrl_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit ReleaseComplianceCtrl_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__ReleaseComplianceCtrl_Request
    std::shared_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__ReleaseComplianceCtrl_Request
    std::shared_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ReleaseComplianceCtrl_Request_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const ReleaseComplianceCtrl_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ReleaseComplianceCtrl_Request_

// alias to use template instance with default allocator
using ReleaseComplianceCtrl_Request =
  dsr_msgs2::srv::ReleaseComplianceCtrl_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__ReleaseComplianceCtrl_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__ReleaseComplianceCtrl_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct ReleaseComplianceCtrl_Response_
{
  using Type = ReleaseComplianceCtrl_Response_<ContainerAllocator>;

  explicit ReleaseComplianceCtrl_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit ReleaseComplianceCtrl_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__ReleaseComplianceCtrl_Response
    std::shared_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__ReleaseComplianceCtrl_Response
    std::shared_ptr<dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ReleaseComplianceCtrl_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const ReleaseComplianceCtrl_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ReleaseComplianceCtrl_Response_

// alias to use template instance with default allocator
using ReleaseComplianceCtrl_Response =
  dsr_msgs2::srv::ReleaseComplianceCtrl_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct ReleaseComplianceCtrl
{
  using Request = dsr_msgs2::srv::ReleaseComplianceCtrl_Request;
  using Response = dsr_msgs2::srv::ReleaseComplianceCtrl_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__RELEASE_COMPLIANCE_CTRL__STRUCT_HPP_
