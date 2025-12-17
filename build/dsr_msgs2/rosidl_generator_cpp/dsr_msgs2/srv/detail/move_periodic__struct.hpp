// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/MovePeriodic.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_PERIODIC__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_PERIODIC__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__MovePeriodic_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__MovePeriodic_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct MovePeriodic_Request_
{
  using Type = MovePeriodic_Request_<ContainerAllocator>;

  explicit MovePeriodic_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->amp.begin(), this->amp.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->periodic.begin(), this->periodic.end(), 0.0);
      this->acc = 0.0;
      this->repeat = 0;
      this->ref = 0;
      this->sync_type = 0;
    }
  }

  explicit MovePeriodic_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : amp(_alloc),
    periodic(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->amp.begin(), this->amp.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->periodic.begin(), this->periodic.end(), 0.0);
      this->acc = 0.0;
      this->repeat = 0;
      this->ref = 0;
      this->sync_type = 0;
    }
  }

  // field types and members
  using _amp_type =
    std::array<double, 6>;
  _amp_type amp;
  using _periodic_type =
    std::array<double, 6>;
  _periodic_type periodic;
  using _acc_type =
    double;
  _acc_type acc;
  using _repeat_type =
    int8_t;
  _repeat_type repeat;
  using _ref_type =
    int8_t;
  _ref_type ref;
  using _sync_type_type =
    int8_t;
  _sync_type_type sync_type;

  // setters for named parameter idiom
  Type & set__amp(
    const std::array<double, 6> & _arg)
  {
    this->amp = _arg;
    return *this;
  }
  Type & set__periodic(
    const std::array<double, 6> & _arg)
  {
    this->periodic = _arg;
    return *this;
  }
  Type & set__acc(
    const double & _arg)
  {
    this->acc = _arg;
    return *this;
  }
  Type & set__repeat(
    const int8_t & _arg)
  {
    this->repeat = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }
  Type & set__sync_type(
    const int8_t & _arg)
  {
    this->sync_type = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__MovePeriodic_Request
    std::shared_ptr<dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__MovePeriodic_Request
    std::shared_ptr<dsr_msgs2::srv::MovePeriodic_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MovePeriodic_Request_ & other) const
  {
    if (this->amp != other.amp) {
      return false;
    }
    if (this->periodic != other.periodic) {
      return false;
    }
    if (this->acc != other.acc) {
      return false;
    }
    if (this->repeat != other.repeat) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    if (this->sync_type != other.sync_type) {
      return false;
    }
    return true;
  }
  bool operator!=(const MovePeriodic_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MovePeriodic_Request_

// alias to use template instance with default allocator
using MovePeriodic_Request =
  dsr_msgs2::srv::MovePeriodic_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__MovePeriodic_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__MovePeriodic_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct MovePeriodic_Response_
{
  using Type = MovePeriodic_Response_<ContainerAllocator>;

  explicit MovePeriodic_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit MovePeriodic_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__MovePeriodic_Response
    std::shared_ptr<dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__MovePeriodic_Response
    std::shared_ptr<dsr_msgs2::srv::MovePeriodic_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MovePeriodic_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const MovePeriodic_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MovePeriodic_Response_

// alias to use template instance with default allocator
using MovePeriodic_Response =
  dsr_msgs2::srv::MovePeriodic_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct MovePeriodic
{
  using Request = dsr_msgs2::srv::MovePeriodic_Request;
  using Response = dsr_msgs2::srv::MovePeriodic_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_PERIODIC__STRUCT_HPP_
