// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/MoveSpiral.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__MoveSpiral_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__MoveSpiral_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct MoveSpiral_Request_
{
  using Type = MoveSpiral_Request_<ContainerAllocator>;

  explicit MoveSpiral_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->revolution = 0.0;
      this->max_radius = 0.0;
      this->max_length = 0.0;
      std::fill<typename std::array<double, 3>::iterator, double>(this->target_pos.begin(), this->target_pos.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->vel.begin(), this->vel.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->acc.begin(), this->acc.end(), 0.0);
      this->time = 0.0;
      this->task_axis = 0;
      this->ref = 0;
      this->mode = 0;
      this->spiral_dir = 0;
      this->rot_dir = 0;
      this->sync_type = 0;
    }
  }

  explicit MoveSpiral_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : target_pos(_alloc),
    vel(_alloc),
    acc(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->revolution = 0.0;
      this->max_radius = 0.0;
      this->max_length = 0.0;
      std::fill<typename std::array<double, 3>::iterator, double>(this->target_pos.begin(), this->target_pos.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->vel.begin(), this->vel.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->acc.begin(), this->acc.end(), 0.0);
      this->time = 0.0;
      this->task_axis = 0;
      this->ref = 0;
      this->mode = 0;
      this->spiral_dir = 0;
      this->rot_dir = 0;
      this->sync_type = 0;
    }
  }

  // field types and members
  using _revolution_type =
    double;
  _revolution_type revolution;
  using _max_radius_type =
    double;
  _max_radius_type max_radius;
  using _max_length_type =
    double;
  _max_length_type max_length;
  using _target_pos_type =
    std::array<double, 3>;
  _target_pos_type target_pos;
  using _vel_type =
    std::array<double, 2>;
  _vel_type vel;
  using _acc_type =
    std::array<double, 2>;
  _acc_type acc;
  using _time_type =
    double;
  _time_type time;
  using _task_axis_type =
    int8_t;
  _task_axis_type task_axis;
  using _ref_type =
    int8_t;
  _ref_type ref;
  using _mode_type =
    int8_t;
  _mode_type mode;
  using _spiral_dir_type =
    int8_t;
  _spiral_dir_type spiral_dir;
  using _rot_dir_type =
    int8_t;
  _rot_dir_type rot_dir;
  using _sync_type_type =
    int8_t;
  _sync_type_type sync_type;

  // setters for named parameter idiom
  Type & set__revolution(
    const double & _arg)
  {
    this->revolution = _arg;
    return *this;
  }
  Type & set__max_radius(
    const double & _arg)
  {
    this->max_radius = _arg;
    return *this;
  }
  Type & set__max_length(
    const double & _arg)
  {
    this->max_length = _arg;
    return *this;
  }
  Type & set__target_pos(
    const std::array<double, 3> & _arg)
  {
    this->target_pos = _arg;
    return *this;
  }
  Type & set__vel(
    const std::array<double, 2> & _arg)
  {
    this->vel = _arg;
    return *this;
  }
  Type & set__acc(
    const std::array<double, 2> & _arg)
  {
    this->acc = _arg;
    return *this;
  }
  Type & set__time(
    const double & _arg)
  {
    this->time = _arg;
    return *this;
  }
  Type & set__task_axis(
    const int8_t & _arg)
  {
    this->task_axis = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }
  Type & set__mode(
    const int8_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }
  Type & set__spiral_dir(
    const int8_t & _arg)
  {
    this->spiral_dir = _arg;
    return *this;
  }
  Type & set__rot_dir(
    const int8_t & _arg)
  {
    this->rot_dir = _arg;
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
    dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__MoveSpiral_Request
    std::shared_ptr<dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__MoveSpiral_Request
    std::shared_ptr<dsr_msgs2::srv::MoveSpiral_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MoveSpiral_Request_ & other) const
  {
    if (this->revolution != other.revolution) {
      return false;
    }
    if (this->max_radius != other.max_radius) {
      return false;
    }
    if (this->max_length != other.max_length) {
      return false;
    }
    if (this->target_pos != other.target_pos) {
      return false;
    }
    if (this->vel != other.vel) {
      return false;
    }
    if (this->acc != other.acc) {
      return false;
    }
    if (this->time != other.time) {
      return false;
    }
    if (this->task_axis != other.task_axis) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    if (this->spiral_dir != other.spiral_dir) {
      return false;
    }
    if (this->rot_dir != other.rot_dir) {
      return false;
    }
    if (this->sync_type != other.sync_type) {
      return false;
    }
    return true;
  }
  bool operator!=(const MoveSpiral_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MoveSpiral_Request_

// alias to use template instance with default allocator
using MoveSpiral_Request =
  dsr_msgs2::srv::MoveSpiral_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__MoveSpiral_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__MoveSpiral_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct MoveSpiral_Response_
{
  using Type = MoveSpiral_Response_<ContainerAllocator>;

  explicit MoveSpiral_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit MoveSpiral_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__MoveSpiral_Response
    std::shared_ptr<dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__MoveSpiral_Response
    std::shared_ptr<dsr_msgs2::srv::MoveSpiral_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MoveSpiral_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const MoveSpiral_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MoveSpiral_Response_

// alias to use template instance with default allocator
using MoveSpiral_Response =
  dsr_msgs2::srv::MoveSpiral_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct MoveSpiral
{
  using Request = dsr_msgs2::srv::MoveSpiral_Request;
  using Response = dsr_msgs2::srv::MoveSpiral_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_SPIRAL__STRUCT_HPP_
